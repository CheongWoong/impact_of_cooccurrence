import os
from tqdm.auto import tqdm
from collections import defaultdict
from copy import deepcopy
import json

import numpy as np

from src.utils.text_processing import text_normalization_without_lemmatization


def get_masks_for_baseline(tokenizer, f_all):
    # generate the gold object mask to restrict candidate sets.
    gold_obj_ids = set()
    gold_obj_relation_wise_ids = defaultdict(set)
    subj_rel_pair_gold_obj_ids = defaultdict(set)

    for example in f_all:
        subj = example['subj']
        rel = example['rel_id']
        obj = example['output']

        obj_id = tokenizer.encode(' '+obj, add_special_tokens=False)[0]
        gold_obj_relation_wise_ids[rel].add(obj_id)
        subj_rel_pair_gold_obj_ids[f'{subj}_{rel}'].add(obj_id)
        gold_obj_ids.add(obj_id)

    ## compute negated ids (== words that are not gold objects)
    gold_obj_mask = [i for i in range(tokenizer.vocab_size)]
    gold_obj_relation_wise_mask = {}

    for gold_obj_id in gold_obj_ids:
        if gold_obj_id in gold_obj_mask:
            gold_obj_mask.remove(gold_obj_id)
    for rel in gold_obj_relation_wise_ids:
        gold_obj_relation_wise_mask[rel] = [i for i in range(tokenizer.vocab_size)]
        for gold_obj_id in gold_obj_relation_wise_ids[rel]:
            gold_obj_relation_wise_mask[rel].remove(gold_obj_id)

    ## set => list
    for key in subj_rel_pair_gold_obj_ids:
        subj_rel_pair_gold_obj_ids[key] = list(subj_rel_pair_gold_obj_ids[key])

    return gold_obj_mask, gold_obj_relation_wise_mask, subj_rel_pair_gold_obj_ids

def postprocess_single_prediction_for_baseline(logits, logits_for_hits_k, tokenizer, label_id):
    results = {}

    # compute top 100 predictions
    sorted_idx = np.argsort(logits)[::-1]
    top_100_idx = sorted_idx[:100]
    results["top_100_text"] = [tokenizer.decode(token_id).strip() for token_id in top_100_idx]
    results["top_100_logits"] = logits[top_100_idx].tolist()
    # compute mrr
    rank = np.where(sorted_idx == label_id)[0][0]+1
    results["mrr"] = 1/rank
    # compute hits@k
    sorted_idx_for_hits_k = np.argsort(logits_for_hits_k)[::-1]
    rank_for_hits_k = np.where(sorted_idx_for_hits_k == label_id)[0][0]+1
    results["hits@1"] = 1.0 if rank_for_hits_k <= 1 else 0.0
    results["hits@10"] = 1.0 if rank_for_hits_k <= 10 else 0.0
    results["hits@100"] = 1.0 if rank_for_hits_k <= 100 else 0.0

    return results

def postprocess_predictions_for_baseline(baseline_type, coo_matrix, validation_dataset, validation_file_path, output_dir, tokenizer):
    # get the masks to restrict output candidate sets.
    with open(os.path.join(os.path.dirname(validation_file_path), 'all.json'), 'r') as fin:
        f_all = json.load(fin)

    gold_obj_mask, gold_obj_relation_wise_mask, subj_rel_pair_gold_obj_ids = get_masks_for_baseline(tokenizer, f_all)

    # get the word indices in the vocab
    vocab = [tokenizer.decode(a).strip() for a in sorted(list(tokenizer.vocab.values()))]

    vocab_obj_idx = []
    for obj in vocab:
        normalized_obj = text_normalization_without_lemmatization(obj)
        if 4 > len(normalized_obj) > 0:
            obj = ' '.join(normalized_obj)
            o_idx = coo_matrix.get_object_idx(obj)
            o_idx = -1 if o_idx is None else o_idx
            vocab_obj_idx.append(o_idx)
        else:
            vocab_obj_idx.append(-1)

    logits_remove_stopwords_marginal = coo_matrix.occurrence_matrix[vocab_obj_idx]
    logits_remove_stopwords_marginal[logits_remove_stopwords_marginal < 0] = 0

    # post-process the predictions for evaluation and save.
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(validation_file_path)
    dataset_name = os.path.basename(os.path.dirname(validation_file_path))
    with open(os.path.join(output_dir, f"pred_{dataset_name}_{basename}l"), "w") as fout:
        print("Processing output predictions...")
        for idx, example in tqdm(enumerate(validation_dataset)):
            subj = example['subj']
            obj = example['output']
            label_text = obj
            label_id = tokenizer.encode(' '+obj, add_special_tokens=False)[0]

            normalized_subj = ' '.join(text_normalization_without_lemmatization(subj))
            subj_idx = coo_matrix.get_subject_idx(normalized_subj)
            subj_idx = -1 if subj_idx is None else subj_idx

            ## 1. remove stopwords
            if baseline_type == 'marginal':
                logits_remove_stopwords = logits_remove_stopwords_marginal
            elif baseline_type == 'joint':
                logits_remove_stopwords = coo_matrix.cooccurrence_matrix[subj_idx, vocab_obj_idx]
            elif baseline_type == 'pmi':
                logits_remove_stopwords = (coo_matrix.cooccurrence_matrix[subj_idx, vocab_obj_idx] + 1) / (logits_remove_stopwords_marginal + 1)
            else:
                raise Exception
            ## 2. 1 + restrict candidates to the set of gold objects in the whole dataset
            logits_gold_objs = logits_remove_stopwords.copy()
            logits_gold_objs[gold_obj_mask] = -10000.
            ## 3. 1 + restrict candidates to the set of gold objects with the same relation
            logits_gold_objs_relation_wise = logits_remove_stopwords.copy()
            logits_gold_objs_relation_wise[gold_obj_relation_wise_mask[example['rel_id']]] = -10000.

            ## When computing hits@1, remove other gold objects for the given subj-rel pair.
            subj_rel_pair_gold_obj_mask = deepcopy(subj_rel_pair_gold_obj_ids[example['subj']+'_'+example['rel_id']])
            subj_rel_pair_gold_obj_mask.remove(label_id)

            logits_for_hits_1_remove_stopwords = logits_remove_stopwords.copy()
            logits_for_hits_1_gold_objs = logits_gold_objs.copy()
            logits_for_hits_1_gold_objs_relation_wise = logits_gold_objs_relation_wise.copy()

            logits_for_hits_1_remove_stopwords[subj_rel_pair_gold_obj_mask] = -10000.
            logits_for_hits_1_gold_objs[subj_rel_pair_gold_obj_mask] = -10000.
            logits_for_hits_1_gold_objs_relation_wise[subj_rel_pair_gold_obj_mask] = -10000.

            ### Compute the results (top 100 predictions, MRR, hits@1)
            postprocessed_results_remove_stopwords = postprocess_single_prediction_for_baseline(logits_remove_stopwords, logits_for_hits_1_remove_stopwords, tokenizer, label_id)
            postprocessed_results_gold_objs = postprocess_single_prediction_for_baseline(logits_gold_objs, logits_for_hits_1_gold_objs, tokenizer, label_id)
            postprocessed_results_gold_objs_relation_wise = postprocess_single_prediction_for_baseline(logits_gold_objs_relation_wise, logits_for_hits_1_gold_objs_relation_wise, tokenizer, label_id)

            postprocessed_results_aggregated = {
                "uid": example["uid"],
                "label_text": label_text,
            }

            for key in postprocessed_results_remove_stopwords:
                postprocessed_results_aggregated[f"{key}_remove_stopwords"] = postprocessed_results_remove_stopwords[key]
            for key in postprocessed_results_gold_objs:
                postprocessed_results_aggregated[f"{key}_gold_objs"] = postprocessed_results_gold_objs[key]
            for key in postprocessed_results_gold_objs_relation_wise:
                postprocessed_results_aggregated[f"{key}_gold_objs_relation_wise"] = postprocessed_results_gold_objs_relation_wise[key]

            json.dump(postprocessed_results_aggregated, fout)
            fout.write('\n')