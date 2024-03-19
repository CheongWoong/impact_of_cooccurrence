from tqdm.auto import tqdm
import jsonlines
import json
import os
import argparse

from nltk import everygrams

from src.utils.text_processing import text_normalization_without_lemmatization

MAX_TOKENS = 2048


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretraining_dataset_name', type=str, default='pile')
    parser.add_argument('--filename', type=str, required=True)
    args = parser.parse_args()

    with open(f"data_statistics/entity_set/merged/all_subjects.json", "r") as fin:
        subject_idx = json.load(fin)
    with open(f"data_statistics/entity_set/merged/all_objects.json", "r") as fin:
        object_idx = json.load(fin)
    all_entities = set(subject_idx.keys()) | set(object_idx.keys())

    out_path = f'data_statistics/term_document_index/{args.pretraining_dataset_name}'
    os.makedirs(out_path, exist_ok=True)

    doc_count, truncated_doc_count = 0, 0 # cwkang: count docs
    pretraining_dataset_path = os.path.join('data', args.pretraining_dataset_name, f'{args.filename}.jsonl')
    with jsonlines.open(pretraining_dataset_path) as fin:
        with open(os.path.join(out_path, f'{args.filename}.jsonl'), 'w') as fout:
            for doc in tqdm(fin.iter()):
                doc_count += 1 # cwkang: count docs

                doc_tokens = text_normalization_without_lemmatization(doc['text'])
                for i in range(0, len(doc_tokens), MAX_TOKENS):
                    truncated_doc_count += 1 # cwkang: count docs

                    truncated_doc = doc_tokens[i:i+MAX_TOKENS]
                    ngrams = set(everygrams(truncated_doc, min_len=1, max_len=3))

                    entities = []
                    for ngram in ngrams:
                        entity = ' '.join(ngram)
                        if entity in all_entities and entity not in entities:
                            entities.append(entity)
                    
                    json.dump({'doc_id': truncated_doc_count, 'entities': entities}, fout)
                    fout.write('\n')

    print(f'number of documents: {doc_count}')
    print(f'number of truncated documents: {truncated_doc_count}')
    print('BYE')