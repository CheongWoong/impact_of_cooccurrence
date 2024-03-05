import argparse
import os

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--pretraining_dataset_name', type=str, default='pile')
    args = parser.parse_args()

    filepath = f'data_statistics/occurrence_matrix/{args.pretraining_dataset_name}'
    filenames = os.listdir(filepath)
    
    occurrence_matrix = None
    for filename in filenames:
        temp_matrix = np.load(os.path.join(filepath, filename))
        if occurrence_matrix is None:
            occurrence_matrix = temp_matrix
        else:
            occurrence_matrix = occurrence_matrix + temp_matrix

    np.save(os.path.join(filepath, 'occurrence_matrix.npy'), occurrence_matrix)
    print("BYE")