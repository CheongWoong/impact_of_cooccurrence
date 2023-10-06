import argparse

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset_name', type=str, default='LAMA_TREx')
    args = parser.parse_args()

    cooccurrence_matrix = np.load(f'data_statistics/cooccurrence_matrix/{args.dataset_name}/00.npy')
    for i in range(1, 30):
        temp_matrix = np.load(f'data_statistics/cooccurrence_matrix/{args.dataset_name}/%02d.npy' % i)
    cooccurrence_matrix = cooccurrence_matrix + temp_matrix

    np.save(f'data_statistics/cooccurrence_matrix/{args.dataset_name}/cooccurrence_matrix.npy', cooccurrence_matrix)
    print("BYE")