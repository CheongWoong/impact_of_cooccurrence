pretraining_dataset_name=$1
dataset_name=$2

python -m src.mitigation.debiasing.debiasing_with_undersampling --pretraining_dataset_name $pretraining_dataset_name --dataset_name $dataset_name