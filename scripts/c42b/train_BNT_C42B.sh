#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=3

cd ../..
python main.py \
\
--model "BNT" \
--within_subject \
--num_repeat 5 \
--subject_num 9 \
\
--dataset 'C42B' \
--data_dir "../data/C42B/C42B128.npy" \
--batch_size 32 \
--num_epochs 200 \
--drop_last True \
--model_dir "output_dir" \
--schedule 'cos' \
--learning_rate 1e-3 \
\
--do_train \
--do_evaluate \
--do_test
