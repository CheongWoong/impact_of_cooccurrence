import numpy as np
import json


class CooccurrenceMatrix():
    def __init__(self):
        with open("../data_statistics/entity_set/LAMA_TREx/entities_with_target_vocab.json", "r") as fin:
            self.entity_idx = json.load(fin)

        self.cooccurrence_matrix = np.load('../data_statistics/cooccurrence_matrix/LAMA_TREx/cooccurrence_matrix.npy')

    def count(self, word):
        idx = self.get_entity_idx(word)
        if idx is not None:
            return self.cooccurrence_matrix[idx][idx].item()
        else:
            return -1

    def coo_count(self, word1, word2):
        idx1 = self.get_entity_idx(word1)
        idx2 = self.get_entity_idx(word2)
        if idx1 is not None and idx2 is not None:
            return self.cooccurrence_matrix[idx1][idx2].item()
        else:
            return -1
    
    def get_entity_idx(self, word):
        return self.entity_idx.get(word, None)

if __name__ == '__main__':
    print("BYE")