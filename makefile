rodar:
	python3 impute.py \
    	-input result_aux_huge.txt \
		-verbose 2 \
		-hidden_size 512 \
		-rel_mask 0.5 \
		-length 150 \
		-offset $(offset) \
		-max_length 25 \
		-it 70000 \
		-eval_ev 1000 \
		-verbose 2 \
		-ilang_size 6 \
		-olang_size 4