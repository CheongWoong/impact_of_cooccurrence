pretraining_dataset_name=$1
filename=$2

OMP_NUM_THREADS=1 python -m src.data_statistics.precompute.compute_cooccurrence_matrix --pretraining_dataset_name $pretraining_dataset_name --filename $filename