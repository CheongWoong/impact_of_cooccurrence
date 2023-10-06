import json
import argparse

from transformers import AutoTokenizer

from src.utils.cooccurrence_matrix import CooccurrenceMatrix
from src.utils.evaluation import postprocess_predictions_for_baseline


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset_name', type=str, default='LAMA_TREx')
    parser.add_argument('--baseline_type', type=str)
    args = parser.parse_args()

    # Initialize the co-occurrence matrix
    coo_matrix = CooccurrenceMatrix(args.dataset_name)

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125m')
    
    validation_file_paths = [f"data/{args.dataset_name}/test.json", f"data/{args.dataset_name}/train.json"]
    for validation_file_path in validation_file_paths:
        validation_dataset = json.load(open(validation_file_path, 'r'))
        output_dir = f'results/{args.baseline_type}'
        postprocess_predictions_for_baseline(args.baseline_type, coo_matrix, validation_dataset, validation_file_path, output_dir, tokenizer)
