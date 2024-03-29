from collections import defaultdict
import argparse
import json

import random
import numpy as np

from src.utils.text_processing import text_normalization_without_lemmatization
from src.utils.cooccurrence_matrix import CooccurrenceMatrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretraining_dataset_name', type=str, default='pile')
    parser.add_argument('--dataset_name', type=str)
    args = parser.parse_args()

    coo_matrix = CooccurrenceMatrix(args.pretraining_dataset_name)

    with open(f"data/{args.dataset_name}/train.json", 'r') as fin:
        f_train = json.load(fin)

    uid_prob_per_rel = defaultdict(list)

    for example in f_train:
        rel = example['rel_id']
        subj = example['subj']
        obj = example['output']
        subj = ' '.join(text_normalization_without_lemmatization(subj))
        obj = ' '.join(text_normalization_without_lemmatization(obj))

        subj_count = coo_matrix.count(subj)
        obj_count = coo_matrix.count(obj)
        subj_obj_count = coo_matrix.coo_count(subj, obj)

        cond_prob = subj_obj_count / subj_count if subj_count > 0 else 0

        uid_prob_per_rel[rel].append((example['uid'], cond_prob))    
    
    for rel in uid_prob_per_rel:
        uid_prob_per_rel[rel] = sorted(uid_prob_per_rel[rel], key=lambda x: x[1], reverse=True)

    filtering_ratios = [0.1, 0.3, 0.5]

    random.seed(0)
    np.random.seed(0)

    for filtering_ratio in filtering_ratios:
        random_filtered_uids = []
        condprob_filtered_uids = []

        for rel in uid_prob_per_rel:
            random_filtered_idx = np.random.choice(range(len(uid_prob_per_rel[rel])), size=int(len(uid_prob_per_rel[rel])*filtering_ratio), replace=False)
            condprob_filtered_idx = list(range(int(len(uid_prob_per_rel[rel])*filtering_ratio)))

            for idx, uid_prob in enumerate(uid_prob_per_rel[rel]):
                if idx not in random_filtered_idx:
                    random_filtered_uids.append(uid_prob[0])
                if idx not in condprob_filtered_idx:
                    condprob_filtered_uids.append(uid_prob[0])
                
        random_filtered_dataset = []
        condprob_filtered_dataset = []

        for example in f_train:
            uid = example['uid']
            if uid in random_filtered_uids:
                random_filtered_dataset.append(example)
            if uid in condprob_filtered_uids:
                condprob_filtered_dataset.append(example)

        with open(f"data/{args.dataset_name}/train_{args.pretraining_dataset_name}_random_filtered_" + str(filtering_ratio) + ".json", "w") as fout:
            json.dump(random_filtered_dataset, fout)
        with open(f"data/{args.dataset_name}/train_{args.pretraining_dataset_name}_debiased_" + str(filtering_ratio) + ".json", "w") as fout:
            json.dump(condprob_filtered_dataset, fout)