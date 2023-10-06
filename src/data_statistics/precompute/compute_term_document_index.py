from tqdm.auto import tqdm
import jsonlines
import json
import os
import argparse

from nltk import everygrams

from src.utils.common.text_processing import text_normalization_without_lemmatization

MAX_LEN = 50000


class DocumentIndex(object):
    def __init__(self, entities):
        self.entities = entities

    def get_doc_entities(self, doc, doc_id):
        output = {
            'doc_id': doc_id,
            'entities': []
        }

        doc = text_normalization_without_lemmatization(doc)
        doc = set(everygrams(doc, min_len=1, max_len=3))

        for ngrams in doc:
            entity = ' '.join(ngrams)
            if entity in self.entities:
                output['entities'].append(entity)

        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pile_path', type=str, default='data/pile/train')
    parser.add_argument('--dataset_name', type=str, default='LAMA_TREx')
    parser.add_argument('--num', type=str, required=True)
    args = parser.parse_args()

    nodes = []
    with open(f'data_statistics/entity_set/{args.dataset_name}/entities_with_target_vocab.json', 'r') as fin:
        entities = json.load(fin)
    document_index = DocumentIndex(entities)

    out_path = f'data_statistics/term_document_index/{args.dataset_name}'
    os.makedirs(out_path, exist_ok=True)
    with jsonlines.open(os.path.join(args.pile_path, f'{args.num}.jsonl')) as fin:
        with open(os.path.join(out_path, f'{args.num}.jsonl'), 'w') as fout:
            doc_count = 0
            for docs in tqdm(fin.iter()):
                docs_text = docs["text"]

                start_idx, end_idx = 0, 0

                while end_idx >= 0:
                    end_idx = docs_text.find(" ", start_idx + MAX_LEN)
                    doc = docs_text[start_idx:end_idx]
                    try:
                        doc_entities = document_index.get_doc_entities(doc, doc_count)
                        json.dump(doc_entities, fout)
                        fout.write('\n')
                        doc_count += 1
                    except Exception as e:
                        print(e)
                    start_idx = end_idx