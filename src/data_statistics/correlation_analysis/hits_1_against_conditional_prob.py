from collections import defaultdict
from tqdm.auto import tqdm
import json
import argparse

import numpy as np

from src.utils.common.text_processing import text_normalization_without_lemmatization
from src.utils.cooccurrence_matrix import CooccurrenceMatrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str)
    parser.add_argument('--dataset_name', type=str, default='LAMA_TREx')
    args = parser.parse_args()

    coo_matrix = CooccurrenceMatrix(args.dataset_name)

    with open(f"data/{args.dataset_name}/all.json", 'r') as fin:
        f_all = json.load(fin)

    uid_rel_map, uid_subj_map, uid_obj_map = {}, {}, {}
    for example in f_all:
        uid_rel_map[example['uid']] = example['rel_id']
        uid_subj_map[example['uid']] = example['subj']
        uid_obj_map[example['uid']] = example['output']

    num_sections = 8

    def prob_value_to_section(value):
        return min(int(np.ceil(-np.log2(value+0.000001))), num_sections - 1)
        
    def prob_section_to_string(section):
        denominator = str(2**section) if section < num_sections - 1 else 'inf'
        return '1/'+denominator

    with open(args.pred_file, "r") as fin:
        preds = json.load(fin)
    results_remove_stopwords, results_gold_objs, results_gold_objs_relation_wise = defaultdict(list), defaultdict(list), defaultdict(list)
    rel_results_remove_stopwords, rel_results_gold_objs, rel_results_gold_objs_relation_wise = defaultdict(dict), defaultdict(dict), defaultdict(dict)

    openai_api = True if 'hits@1_gold_objs' not in preds[0] else False

    for pred in tqdm(preds):
        rel = uid_rel_map[pred['uid']]
        subj = uid_subj_map[pred['uid']]
        obj = uid_obj_map[pred['uid']]
        subj = ' '.join(text_normalization_without_lemmatization(subj))
        obj = ' '.join(text_normalization_without_lemmatization(obj))

        subj_count = coo_matrix.count(subj)
        obj_count = coo_matrix.count(obj)
        subj_obj_count = coo_matrix.coo_count(subj, obj)

        if subj_obj_count < 0:
            continue

        obj_prob = obj_count / 210000000
        joint_prob = subj_obj_count / 210000000
        cond_prob = subj_obj_count / subj_count if subj_count > 0 else 0

        prob = cond_prob

        section_prob_remove_stopwords = prob_value_to_section(prob)
        section_prob_gold_objs = prob_value_to_section(prob)
        section_prob_gold_objs_relation_wise = prob_value_to_section(prob)

        results_remove_stopwords[section_prob_remove_stopwords].append(pred['hits@1_remove_stopwords'])
        if not openai_api:
            results_gold_objs[section_prob_gold_objs].append(pred['hits@1_gold_objs'])
            results_gold_objs_relation_wise[section_prob_gold_objs_relation_wise].append(pred['hits@1_gold_objs_relation_wise'])

        if section_prob_remove_stopwords in rel_results_remove_stopwords[rel]:
            rel_results_remove_stopwords[rel][section_prob_remove_stopwords].append(pred['hits@1_remove_stopwords'])
        else:
            rel_results_remove_stopwords[rel][section_prob_remove_stopwords] = [pred['hits@1_remove_stopwords']]
        if not openai_api:
            if section_prob_gold_objs in rel_results_gold_objs[rel]:
                rel_results_gold_objs[rel][section_prob_gold_objs].append(pred['hits@1_gold_objs'])
            else:
                rel_results_gold_objs[rel][section_prob_gold_objs] = [pred['hits@1_gold_objs']]
            if section_prob_gold_objs_relation_wise in rel_results_gold_objs_relation_wise[rel]:
                rel_results_gold_objs_relation_wise[rel][section_prob_gold_objs_relation_wise].append(pred['hits@1_gold_objs_relation_wise'])
            else:
                rel_results_gold_objs_relation_wise[rel][section_prob_gold_objs_relation_wise] = [pred['hits@1_gold_objs_relation_wise']]

    sections = range(num_sections)
    for section in sections:
        print(section, len(results_remove_stopwords[section]), len(results_gold_objs[section]), len(results_gold_objs_relation_wise[section])) #####
        results_remove_stopwords[section] = np.mean(results_remove_stopwords[section]), np.std(results_remove_stopwords[section])
        if not openai_api:
            results_gold_objs[section] = np.mean(results_gold_objs[section]), np.std(results_gold_objs[section])
            results_gold_objs_relation_wise[section] = np.mean(results_gold_objs_relation_wise[section]), np.std(results_gold_objs_relation_wise[section])

    for rel in rel_results_remove_stopwords:
        for section in sections:
            if section in rel_results_remove_stopwords[rel]:
                rel_results_remove_stopwords[rel][section] = np.mean(rel_results_remove_stopwords[rel][section]), np.std(rel_results_remove_stopwords[rel][section])
            else:
                rel_results_remove_stopwords[rel][section] = -1.0, -1.0
            if not openai_api:
                if section in rel_results_gold_objs[rel]:
                    rel_results_gold_objs[rel][section] = np.mean(rel_results_gold_objs[rel][section]), np.std(rel_results_gold_objs[rel][section])
                else:
                    rel_results_gold_objs[rel][section] = -1.0, -1.0
                if section in rel_results_gold_objs_relation_wise[rel]:
                    rel_results_gold_objs_relation_wise[rel][section] = np.mean(rel_results_gold_objs_relation_wise[rel][section]), np.std(rel_results_gold_objs_relation_wise[rel][section])
                else:
                    rel_results_gold_objs_relation_wise[rel][section] = -1.0, -1.0

    result = {}
    for section in sections:
        result['hits@1_remove_stopwords_section_' + prob_section_to_string(section)] = f"%.2f +- %.2f" % results_remove_stopwords[section]
    if not openai_api:
        for section in sections:
            result['hits@1_gold_objs_section_' + prob_section_to_string(section)] = f"%.2f +- %.2f" % results_gold_objs[section]
        for section in sections:
            result['hits@1_gold_objs_relation_wise_section_' + prob_section_to_string(section)] = f"%.2f +- %.2f" % results_gold_objs_relation_wise[section]

    sorted_rels = sorted(list(rel_results_remove_stopwords.keys()))
    for rel in sorted_rels:
        for section in sections:
            result['hits@1_remove_stopwords_' + rel + '_section_' + prob_section_to_string(section)] = f"%.2f +- %.2f" % rel_results_remove_stopwords[rel][section]

    if not openai_api:
        for rel in sorted_rels:
            for section in sections:
                result['hits@1_gold_objs_' + rel + '_section_' + prob_section_to_string(section)] = f"%.2f +- %.2f" % rel_results_gold_objs[rel][section]

        for rel in sorted_rels:
            for section in sections:
                result['hits@1_gold_objs_relation_wise_' + rel + '_section_' + prob_section_to_string(section)] = f"%.2f +- %.2f" % rel_results_gold_objs_relation_wise[rel][section]

    with open(args.pred_file.replace('pred', 'hits_1_against_conditional_prob'), 'w') as fout:
        json.dump(result, fout, indent=4)