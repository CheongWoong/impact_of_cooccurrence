from tqdm.auto import tqdm
import json
import os
import argparse
import re

from transformers import AutoTokenizer

from src.utils.text_processing import text_normalization_without_lemmatization


def is_pure_alphanumeric(token):
	return re.match("^[a-zA-Z0-9]+$", token) is not None

def get_entities_from_dataset(dataset_path):
	subjects = set()
	objects = set()
	entities = set()

	with open(dataset_path, 'r') as fin:
		data = json.load(fin)
		for idx, sample in tqdm(enumerate(data)):
			subj = sample['subj']
			normalized_subj = text_normalization_without_lemmatization(subj)
			if 4 > len(normalized_subj) > 0:
				subj = ' '.join(normalized_subj)
				subjects.add(subj)
				entities.add(subj)

			obj = sample['output']
			normalized_obj = text_normalization_without_lemmatization(obj)
			if (4 > len(normalized_obj) > 0):
				obj = ' '.join(normalized_obj)
				objects.add(obj)
				entities.add(obj)

	subjects = sorted(subjects)
	objects = sorted(objects)
	entities = sorted(entities)

	subjects_inverted_idx = {}
	for idx, w in enumerate(subjects):
		subjects_inverted_idx[w] = idx

	objects_inverted_idx = {}
	for idx, w in enumerate(objects):
		objects_inverted_idx[w] = idx

	entities_inverted_idx = {}
	for idx, w in enumerate(entities):
		entities_inverted_idx[w] = idx

	return subjects_inverted_idx, objects_inverted_idx, entities_inverted_idx

def get_entities_from_vocab(model_names):
	target_vocab_entities = set()

	vocab_intersection = set()
	vocab_union = set()
	for model_name in model_names:
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		vocab = set()
		for id in tokenizer.vocab.values():
			word = tokenizer.decode(id).strip()
			if is_pure_alphanumeric(word) and len(tokenizer.encode(' '+word, add_special_tokens=False)) == 1:
				vocab.add(word)
		vocab_intersection = vocab_intersection & vocab if len(vocab_intersection) > 0 else vocab
		vocab_union = vocab_union | vocab

	for word in vocab_union:
		normalized_word = text_normalization_without_lemmatization(word)
		if (4 > len(normalized_word) > 0):
			target_vocab_entities.add(' '.join(normalized_word))

	target_vocab_entities = sorted(target_vocab_entities)

	target_vocab_entities_inverted_idx = {}
	for idx, w in enumerate(target_vocab_entities):
		target_vocab_entities_inverted_idx[w] = idx

	return target_vocab_entities_inverted_idx


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
        '--model_names',
        nargs='+',
        default=[
			'EleutherAI/gpt-neo-125m', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6b',
			'bert-base-uncased', 'bert-large-uncased',
			'roberta-base', 'roberta-large',
			'albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1',
			'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2',
			'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-8B-Instruct'
		]
	)
	parser.add_argument(
		'--dataset_names',
		nargs='+',
		default=[
			'LAMA_TREx', 'ConceptNet'
		]
	)
	args = parser.parse_args()
	
	all_subjects = set()
	all_objects = set()
	all_entities = set()

	### Extract entity set from datasets
	for dataset_name in args.dataset_names:
		out_path = f'data_statistics/entity_set/{dataset_name}'
		
		if not os.path.exists(out_path):
			data_path = f'data/{dataset_name}/all.json'
			subjects, objects, entities = get_entities_from_dataset(data_path)

			os.makedirs(out_path)
			with open(os.path.join(out_path, 'subjects.json'), 'w') as fout:
				json.dump(subjects, fout)
			with open(os.path.join(out_path, 'objects.json'), 'w') as fout:
				json.dump(objects, fout)
			with open(os.path.join(out_path, 'entities.json'), 'w') as fout:
				json.dump(entities, fout)
		else:
			with open(os.path.join(out_path, 'subjects.json'), 'r') as fin:
				subjects = json.load(fin)
			with open(os.path.join(out_path, 'objects.json'), 'r') as fin:
				objects = json.load(fin)
			with open(os.path.join(out_path, 'entities.json'), 'r') as fin:
				entities = json.load(fin)

		all_subjects = all_subjects | set(subjects.keys())
		all_objects = all_objects | set(objects.keys())
		all_entities = all_entities | set(entities.keys())

		print(f'Number of {dataset_name} subjects:', len(subjects))
		print(f'Number of {dataset_name} objects:', len(objects))
		print(f'Number of {dataset_name} entities:', len(entities))
		print('='*20)

	### Extract entity set from target vocab
	out_path = f'data_statistics/entity_set/target_vocab'

	target_vocab_entities = get_entities_from_vocab(args.model_names)

	os.makedirs(out_path, exist_ok=True)
	with open(os.path.join(out_path, 'entities.json'), 'w') as fout:
		json.dump(target_vocab_entities, fout)

	all_objects = all_objects | set(target_vocab_entities.keys())
	all_entities = all_entities | set(target_vocab_entities.keys())

	print(f'Number of target vocab entities:', len(target_vocab_entities))
	print('='*20)

	### Write the merged entities
	out_path = f'data_statistics/entity_set/merged'
	all_subjects = sorted(all_subjects)
	all_objects = sorted(all_objects)
	all_entities = sorted(all_entities)

	all_subjects_inverted_idx = {}
	for idx, w in enumerate(all_subjects):
		all_subjects_inverted_idx[w] = idx

	all_objects_inverted_idx = {}
	for idx, w in enumerate(all_objects):
		all_objects_inverted_idx[w] = idx

	all_entities_inverted_idx = {}
	for idx, w in enumerate(all_entities):
		all_entities_inverted_idx[w] = idx

	os.makedirs(out_path, exist_ok=True)
	with open(os.path.join(out_path, 'all_subjects.json'), 'w') as fout:
		json.dump(all_subjects_inverted_idx, fout)
	with open(os.path.join(out_path, 'all_objects.json'), 'w') as fout:
		json.dump(all_objects_inverted_idx, fout)
	with open(os.path.join(out_path, 'all_entities.json'), 'w') as fout:
		json.dump(all_entities_inverted_idx, fout)

	print(f'Number of merged subjects:', len(all_subjects))
	print(f'Number of merged objects:', len(all_objects))
	print(f'Number of merged entities:', len(all_entities))
	print('='*20)
