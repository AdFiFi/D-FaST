#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=3
#export WANDB_API_KEY=local-1a6c67774093a56b310d8313b6821f2d98e59678
#export WANDB_MODE=offline

cd ../..
python main.py \
\
--project "FaST-P-C42B1" \
--model "FaSTP" \
--num_repeat 5 \
\
--dataset "C42B" \
--data_dir "../data/C42B/C42B128.npy" \
--sparsity 1 \
--batch_size 32 \
--num_epochs 200 \
--frequency 128 \
--num_kernels 64 \
--window_size 3 \
--D 1 \
--p1 8 \
--p2 16 \
--drop_last True \
--num_heads 4 \
--distill \
--mix_up \
--num_layers 1 \
--learning_rate 1e-3 \
--dropout 0.5 \
--schedule 'cos' \
\
--do_train \
--do_evaluate \
--do_test
