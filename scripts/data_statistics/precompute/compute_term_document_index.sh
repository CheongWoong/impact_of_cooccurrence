pretraining_dataset_name=$1
filename=$2

python -m src.data_statistics.precompute.compute_term_document_index --pretraining_dataset_name $pretraining_dataset_name --filename $filename