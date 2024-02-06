from collections import defaultdict
import argparse
import json

import random
import numpy as np

from src.utils.text_processing import text_normalization_without_lemmatization
from src.utils.cooccurrence_matrix import CooccurrenceMatrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    args = parser.parse_args()

    coo_matrix = CooccurrenceMatrix(args.dataset_name)

    with open(f"data/{args.dataset_name}/all.json", 'r') as fin:
        f_all = json.load(fin)

    uid_rel_map, uid_subj_map, uid_obj_map = {}, {}, {}
    for example in f_all:
        uid_rel_map[example['uid']] = example['rel_id']
        uid_subj_map[example['uid']] = example['subj']
        uid_obj_map[example['uid']] = example['output']

    with open(f"data/{args.dataset_name}/train.json", 'r') as fin:
        f_train = json.load(fin)

    with open(f"results/joint/pred_{args.dataset_name}_train.json", "r") as fin:
        baseline_train = json.load(fin)

    uid_prob_per_rel = defaultdict(list)

    for baseline in baseline_train:
        rel = uid_rel_map[baseline['uid']]
        subj = uid_subj_map[baseline['uid']]
        obj = uid_obj_map[baseline['uid']]
        subj = ' '.join(text_normalization_without_lemmatization(subj))
        obj = ' '.join(text_normalization_without_lemmatization(obj))

        subj_count = coo_matrix.count(subj)
        obj_count = coo_matrix.count(obj)
        subj_obj_count = coo_matrix.coo_count(subj, obj)

        obj_prob = obj_count / 210000000
        joint_prob = subj_obj_count / 210000000
        cond_prob = subj_obj_count / subj_count if subj_count > 0 else 0

        prob = cond_prob

        uid_prob_per_rel[rel].append((baseline['uid'], prob))    
    
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

        with open(f"data/{args.dataset_name}/train_random_filtered_" + str(filtering_ratio) + ".json", "w") as fout:
            json.dump(random_filtered_dataset, fout)
        with open(f"data/{args.dataset_name}/train_debiased_" + str(filtering_ratio) + ".json", "w") as fout:
            json.dump(condprob_filtered_dataset, fout)