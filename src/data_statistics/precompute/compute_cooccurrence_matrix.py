from tqdm.auto import tqdm
import jsonlines
import json
import os
import argparse

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--pretraining_dataset_name', type=str, default='pile')
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()
    
    with open(f"data_statistics/entity_set/merged/all_subjects.json", "r") as fin:
        subject_idx = json.load(fin)
    with open(f"data_statistics/entity_set/merged/all_objects.json", "r") as fin:
        object_idx = json.load(fin)

    def get_subject_idx(entity):
        return subject_idx.get(entity, -1)
    def get_object_idx(entity):
        return object_idx.get(entity, -1)

    mat_shape = (len(subject_idx) + 1, len(object_idx) + 1)
    cooccurrence_matrix = np.zeros(mat_shape, dtype=np.int32)
    cooccurrence_matrix *= 0

    term_document_index_path = os.path.join('data_statistics', 'term_document_index', args.pretraining_dataset_name, f'{args.filename}.jsonl')
    with jsonlines.open(term_document_index_path) as fin:
        for line in tqdm(fin.iter()):
            entities = line['entities']

            if len(entities) > 0:
                s_idx = np.fromiter(map(get_subject_idx, entities), dtype=np.int32)
                o_idx = np.fromiter(map(get_object_idx, entities), dtype=np.int32)

                cooccurrence_matrix[s_idx[:,None], o_idx[None,:]] += 1

    out_path = f'data_statistics/cooccurrence_matrix/{args.pretraining_dataset_name}'
    os.makedirs(out_path, exist_ok=True)
    np.save(os.path.join(out_path, f'{args.filename}.npy'), cooccurrence_matrix)

    print('BYE')