from tqdm.auto import tqdm
import jsonlines
import json
import os
import argparse

from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer

from src.utils.common.text_processing import text_normalization_without_lemmatization


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names', nargs='+', default=['EleutherAI/gpt-neo-125m', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6b'])
    args = parser.parse_args()

    vocab_intersection = set()
    vocab_union = set()
    for model_name in args.model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        vocab = set([token for token in list(tokenizer.vocab.keys())])
        vocab_intersection = vocab_intersection & vocab if len(vocab_intersection) > 0 else vocab
        vocab_union = vocab_union | vocab

    relations = {}
    with jsonlines.open('data/original_LAMA/data/relations.jsonl') as fin:
        for rel in fin.iter():
            relations[rel['relation']] = rel['template']

    trex_path = 'data/original_LAMA/data/TREx'
    data_paths = os.listdir(trex_path)
    for i in range(len(data_paths)):
        data_paths[i] = os.path.join(trex_path, data_paths[i])

    prompts = []
    for data_path in data_paths:
        with jsonlines.open(data_path) as fin:
            for idx, sample in tqdm(enumerate(fin.iter())):
                uid = sample['uuid']
                subj = sample['sub_label'].strip()
                obj = sample['obj_label'].strip()
                rel_id = sample['predicate_id']
                if 'Ġ' + obj not in vocab_intersection:
                    continue
                if not (2 > len(text_normalization_without_lemmatization(obj)) > 0):
                    continue

                template = relations[rel_id]
                input_prompt = template.replace('[X]', subj).replace('[Y]', '[MASK]')
                input_prompt_truncated = input_prompt.split('[MASK]')[0].strip()
                output = obj

                prompts.append({
                    'uid': uid,
                    'subj': subj,
                    'rel_id': rel_id,
                    'input': input_prompt,
                    'truncated_input': input_prompt_truncated,
                    'output': output,
                })

    out_path = 'data/LAMA_TREx'
    os.makedirs(out_path, exist_ok=True)
    print(len(prompts))
    train, test = train_test_split(prompts, train_size=0.7, random_state=0)
    with open(os.path.join(out_path, 'train.json'), 'w') as fout:
        json.dump(train, fout)
    with open(os.path.join(out_path, 'test.json'), 'w') as fout:
        json.dump(test, fout)
    with open(os.path.join(out_path, 'all.json'), 'w') as fout:
        json.dump(prompts, fout)