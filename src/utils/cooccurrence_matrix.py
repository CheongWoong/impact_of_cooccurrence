import numpy as np
import json


class CooccurrenceMatrix():
    def __init__(self, pretraining_dataset_name):
        with open(f"data_statistics/entity_set/merged/all_subjects.json", "r") as fin:
            self.subject_idx = json.load(fin)
        with open(f"data_statistics/entity_set/merged/all_objects.json", "r") as fin:
            self.object_idx = json.load(fin)
        with open(f"data_statistics/entity_set/merged/all_entities.json", "r") as fin:
            self.entity_idx = json.load(fin)

        self.cooccurrence_matrix = np.load(f'data_statistics/cooccurrence_matrix/{pretraining_dataset_name}/cooccurrence_matrix.npy')
        self.occurrence_matrix = np.load(f'data_statistics/occurrence_matrix/{pretraining_dataset_name}/occurrence_matrix.npy')

    def count(self, word):
        idx = self.get_entity_idx(word)
        if idx is not None:
            return self.occurrence_matrix[idx].item()
        else:
            return -1

    def coo_count(self, subj, obj):
        s_idx = self.get_subject_idx(subj)
        o_idx = self.get_object_idx(obj)
        if s_idx is not None and o_idx is not None:
            return self.cooccurrence_matrix[s_idx][o_idx].item()
        else:
            return -1
    
    def get_subject_idx(self, word):
        return self.subject_idx.get(word, None)
    
    def get_object_idx(self, word):
        return self.object_idx.get(word, None)

    def get_entity_idx(self, word):
        return self.entity_idx.get(word, None)

if __name__ == '__main__':
    print("BYE")