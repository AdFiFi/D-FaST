#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=2
#export WANDB_API_KEY=local-1a6c67774093a56b310d8313b6821f2d98e59678
#export WANDB_MODE=offline

cd ../..
python main.py \
--wandb_entity cwg \
--project SMR \
\
--model "FaSTP" \
--num_repeat 5 \
\
--dataset 'SMR' \
--data_dir "/data/datasets/SMR/SMR128.npy" \
--sparsity 0.6 \
--batch_size 32 \
--num_epochs 200 \
--frequency 128 \
--num_kernels 64 \
--window_size 32 \
--D 22 \
--p1 4 \
--p2 8 \
--drop_last True \
--num_heads 4 \
--distill \
--num_layers 1 \
--learning_rate 1e-3 \
--dropout 0.5 \
--schedule 'cos' \
\
--do_train \
--do_evaluate \
--do_test
