

MODEL=

python -u \
    --data_dir ../../data/processed/ \
    --vocab_path ../../models/${MODEL}/vocab.txt \
    --output_dir ./outputs/ \ 
    --data_name bert \
    --seed 2020 \
    --max_seq_len 512 \
    --short_seq_prob 0 \
    --masked_lm_prob 0.2 \
    --max_predictions_per_seq 20 


