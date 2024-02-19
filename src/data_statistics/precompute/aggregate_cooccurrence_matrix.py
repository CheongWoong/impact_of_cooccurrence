import argparse
import os

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--pretraining_dataset_name', type=str, default='pile')
    args = parser.parse_args()

    filepath = f'data_statistics/cooccurrence_matrix/{args.pretraining_dataset_name}'
    filenames = os.listdir(filepath)
    
    cooccurrence_matrix = None
    for filename in filenames:
        temp_matrix = np.load(os.path.join(filepath, filename))
        if cooccurrence_matrix is None:
            cooccurrence_matrix = temp_matrix
        else:
            cooccurrence_matrix = cooccurrence_matrix + temp_matrix

    np.save(os.path.join(filepath, 'cooccurrence_matrix.npy'), cooccurrence_matrix)
    print("BYE")