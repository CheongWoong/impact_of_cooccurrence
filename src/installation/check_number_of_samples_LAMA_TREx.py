from collections import defaultdict
import json


if __name__ == '__main__':
    dataset_name = 'LAMA_TREx'

    with open(f'data/{dataset_name}/train.json', 'r') as fin:
        train = json.load(fin)
    with open(f'data/{dataset_name}/test.json', 'r') as fin:
        test = json.load(fin)
    with open(f'data/{dataset_name}/all.json', 'r') as fin:
        all = json.load(fin)

    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    all_counts = defaultdict(int)
    for sample in train:
        train_counts[sample['rel_id']] += 1
        train_counts['all'] += 1
    for sample in test:
        test_counts[sample['rel_id']] += 1
        test_counts['all'] += 1
    for sample in all:
        all_counts[sample['rel_id']] += 1
        all_counts['all'] += 1

    sorted_keys = sorted(list(all_counts.keys()), key=lambda x: int(x[1:]) if x[1:].isdigit() else 10000)

    print("\tTrain / %5s / %5s" % ('Test', 'All'))
    for key in sorted_keys:
        print(f"{key}:\t%5d / %5d / %5d" % (train_counts[key], test_counts[key], all_counts[key]))
    print()