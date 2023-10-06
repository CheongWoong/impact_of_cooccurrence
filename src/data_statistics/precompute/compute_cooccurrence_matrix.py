from tqdm.auto import tqdm
import jsonlines
import json
import os
import argparse

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--num", type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='LAMA_TREx')
    args = parser.parse_args()
    
    with open(f"data_statistics/entity_set/{args.dataset_name}/entities_with_target_vocab.json", "r") as fin:
        entity_idx = json.load(fin)

    def get_entity_idx(entity):
        return entity_idx.get(entity, -1)

    mat_shape = (len(entity_idx) + 1, len(entity_idx) + 1)
    cooccurrence_matrix = np.zeros(mat_shape, dtype=np.int32)
    # cooccurrence_matrix *= 0

    out_path = f'data_statistics/term_document_index/{args.dataset_name}'
    os.makedirs(out_path, exist_ok=True)

    count = 0
    with jsonlines.open(os.path.join(out_path, f'{args.num}.jsonl')) as fin:
        for line in tqdm(fin.iter()):
            entities = line["entities"]

            eidx = np.fromiter(map(get_entity_idx, entities), dtype=np.int64)

            cooccurrence_matrix[eidx[:,None],eidx[None,:]] += 1

    out_path = f'data_statistics/cooccurrence_matrix/{args.dataset_name}'
    os.makedirs(out_path, exist_ok=True)
    np.save(os.path.join(out_path, f'{args.num}.npy'), cooccurrence_matrix)

    print("BYE")