#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=3
#export WANDB_API_KEY=local-1a6c67774093a56b310d8313b6821f2d98e59678

cd ../..
python main.py \
\
--model "FaSTP" \
--num_repeat 5 \
\
--dataset 'BR' \
--data_dir "../data/BR/BR.npy" \
--sparsity 0.6 \
--batch_size 16 \
--num_epochs 100 \
--frequency 200 \
--num_kernels 256 \
--window_size 16 \
--D 16 \
--p1 8 \
--p2 16 \
--drop_last True \
--num_heads 4 \
--distill \
--num_layers 1 \
--learning_rate 1e-4 \
--dropout 0.1 \
--schedule 'cos' \
\
--do_train \
--do_evaluate \
--do_test
