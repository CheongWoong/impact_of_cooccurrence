pretraining_dataset_name=$1

python -m src.data_statistics.precompute.aggregate_occurrence_matrix --pretraining_dataset_name $pretraining_dataset_name