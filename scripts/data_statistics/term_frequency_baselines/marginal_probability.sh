pretraining_dataset_name=$1
dataset_name=$2

python -m src.data_statistics.term_frequency_baselines.term_frequency_baseline --pretraining_dataset_name $pretraining_dataset_name --dataset_name $dataset_name --baseline_type marginal