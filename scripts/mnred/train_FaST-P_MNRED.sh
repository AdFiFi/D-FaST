#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=3

cd ../..
python main.py \
\
--model "DFaST" \
--num_repeat 5 \
\
--dataset 'MNRED' \
--data_dir "../data/MNRED/MNRED.npy" \
--sparsity 0.6 \
--batch_size 16 \
--num_epochs 100 \
--frequency 200 \
--num_kernels 128 \
--window_size 16 \
--D 30 \
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
