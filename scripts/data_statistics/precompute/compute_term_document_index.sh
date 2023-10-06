for NUM in {00..29}
do
	nohup python -m src.data_statistics.precompute.compute_term_document_index --num $NUM > "logs/log.compute_term_document_index_"$NUM &
done