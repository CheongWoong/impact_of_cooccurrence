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
    
    with open(f"data_statistics/entity_set/merged/all_entities.json", "r") as fin:
        entity_idx = json.load(fin)

    def get_entity_idx(entity):
        return entity_idx.get(entity, -1)

    occurrence_matrix = np.zeros(len(entity_idx) + 1, dtype=np.int32)
    # occurrence_matrix *= 0

    term_document_index_path = os.path.join('data_statistics', 'term_document_index', args.pretraining_dataset_name, f'{args.filename}.jsonl')
    with jsonlines.open(term_document_index_path) as fin:
        for line in tqdm(fin.iter()):
            entities = line['entities']

            if len(entities) > 0:
                idx = np.fromiter(map(get_entity_idx, entities), dtype=np.int32)

                occurrence_matrix[idx] += 1

    out_path = f'data_statistics/occurrence_matrix/{args.pretraining_dataset_name}'
    os.makedirs(out_path, exist_ok=True)
    np.save(os.path.join(out_path, f'{args.filename}.npy'), occurrence_matrix)

    print('BYE')