python3 -u run_supervised.py --data_dir ./data/supervised \
--model_type bert \
--labels ./data/labels.txt \
--model_name_or_path ../predata \
--do_lower_case \
--output_dir ./output/ \
--overwrite_output_dir \
--max_seq_length  128 \
--evaluate_during_training \
--logging_steps 2000 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--save_steps 2000 \
--max_steps 20000 \
--seed 1 \
--do_predict \
--do_train \
--do_eval
