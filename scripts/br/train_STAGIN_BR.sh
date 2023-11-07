#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=2
export WANDB_API_KEY=cd4441a5fcdd740b84b45deb6890ecb376bddecb
export WANDB_MODE=offline

cd ../..
python main.py \
\
--model "STAGIN" \
--num_repeat 5 \
\
--dataset 'BR' \
--data_dir "../data/BR/BR.npy" \
--percentage 1. \
--batch_size 16 \
--num_epochs 100 \
--drop_last True \
\
--d_model 64 \
--window_size 50 \
--window_stride 3 \
--dynamic_length 440 \
--num_heads 1 \
--num_layers 2 \
\
--do_train \
\
--learning_rate 0.0005 \
--max_learning_rate 0.001 \
--schedule 'one_cycle' \
\
--do_evaluate \
--do_test