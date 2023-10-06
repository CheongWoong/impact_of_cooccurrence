from tqdm.auto import tqdm
import json
import os
import argparse

from transformers import AutoTokenizer

from src.utils.common.text_processing import text_normalization_without_lemmatization


def get_entities(model_names, dataset_path):
	entities = set()
	subjects = set()
	objects = set()
	all_entities = set()

	### Add entities in the dataset
	with open(dataset_path, 'r') as fin:
		data = json.load(fin)
		for idx, sample in tqdm(enumerate(data)):
			subj = sample['subj']
			obj = sample['output']
			normalized_subj = text_normalization_without_lemmatization(subj)
			normalized_obj = text_normalization_without_lemmatization(obj)

			if (4 > len(normalized_obj) > 0):
				obj = ' '.join(normalized_obj)
				entities.add(obj)
				objects.add(obj)
				### Add all tokens
				for token in normalized_obj:
					entities.add(token)
					objects.add(token)
			if 4 > len(normalized_subj) > 0:
				subj = ' '.join(normalized_subj)
				entities.add(subj)
				subjects.add(subj)
				### Add all tokens
				for token in normalized_subj:
					entities.add(token)
					subjects.add(token)

	all_entities = all_entities | entities
	### Add words in target models' vocab
	vocab_intersection = set()
	vocab_union = set()
	for model_name in model_names:
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		vocab = set([token.replace('Ġ', '') for token in list(tokenizer.vocab.keys()) if 'Ġ' in token])
		vocab_intersection = vocab_intersection & vocab if len(vocab_intersection) > 0 else vocab
		vocab_union = vocab_union | vocab

	for word in vocab_union:
		normalized_word = text_normalization_without_lemmatization(word)
		all_entities.add(' '.join(normalized_word))
		### Add all tokens
		for token in normalized_word:
			all_entities.add(token)

	### Add words in relational templates
	templates = [
		'born', 'died', 'subclass', 'official language', 'plays',
		'awarded', 'originally aired', 'educated university', 'shares border', 'named after',
		'original language', 'plays', 'member', 'works field', 'participated',
		'profession', 'consists', 'member political party', 'maintains diplomatic relations', 'produced',
		'citizen', 'written', 'located', 'developed', 'capital',
		'located', 'communicate', 'works', 'plays music', 'located',
		'position', 'music label', 'located', 'work', 'affiliated religion',
		'plays', 'owned', 'native language', 'twin cities', 'legal term',
		'is', 'created', 'headquarter', 'capital', 'founded',
		'part'
	]
	for template in templates:
		normalized_template = text_normalization_without_lemmatization(template)
		### Add tokens only
		for token in normalized_template:
			all_entities.add(token)

	## Create inverted index
	entities = sorted(entities)
	subjects = sorted(subjects)
	objects = sorted(objects)
	all_entities = sorted(all_entities)

	entities_inverted_idx = {}
	for idx, w in enumerate(entities):
		entities_inverted_idx[w] = idx

	subjects_inverted_idx = {}
	for idx, w in enumerate(subjects):
		subjects_inverted_idx[w] = idx

	objects_inverted_idx = {}
	for idx, w in enumerate(objects):
		objects_inverted_idx[w] = idx

	all_entities_inverted_idx = {}
	for idx, w in enumerate(all_entities):
		all_entities_inverted_idx[w] = idx

	return entities_inverted_idx, subjects_inverted_idx, objects_inverted_idx, all_entities_inverted_idx


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_names', nargs='+', default=['EleutherAI/gpt-neo-125m', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6b'])
	parser.add_argument('--dataset_name', type=str, default='LAMA_TREx')
	args = parser.parse_args()
	
	data_path = os.path.join('data', args.dataset_name, 'all.json')
	entities, subjects, objects, all_entities = get_entities(args.model_names, data_path)

	out_path = f'data_statistics/entity_set/{args.dataset_name}'
	os.makedirs(out_path, exist_ok=True)
	with open(os.path.join(out_path, 'entities.json'), 'w') as fout:
		json.dump(entities, fout)
	with open(os.path.join(out_path, 'subjects.json'), 'w') as fout:
		json.dump(subjects, fout)
	with open(os.path.join(out_path, 'objects.json'), 'w') as fout:
		json.dump(objects, fout)
	with open(os.path.join(out_path, 'entities_with_target_vocab.json'), 'w') as fout:
		json.dump(all_entities, fout)

	print(f'Number of {args.dataset_name} entities:', len(entities))
	print(f'Number of {args.dataset_name} subjects:', len(subjects))
	print(f'Number of {args.dataset_name} objects:', len(objects))
	print(f'Number of {args.dataset_name} and target vocab entities:', len(all_entities))
	print('Done')