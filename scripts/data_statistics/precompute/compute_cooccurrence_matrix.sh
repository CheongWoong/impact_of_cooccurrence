for NUM in {00..29}
do
	nohup python -m src.data_statistics.precompute.compute_cooccurrence_matrix --num $NUM > "logs/log.compute_cooccurrence_matrix_"$NUM &
done