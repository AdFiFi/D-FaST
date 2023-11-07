#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=2
export WANDB_API_KEY=local-1be95a8bdeaf0e5e73d1b448429ff93821d77b3d

cd ../..
python main.py \
--num_repeat 5 \
--wandb_entity cwg \
--project ZuCo-NR \
\
--model "FaSTPOnlySpatial" \
--num_repeat 5 \
\
--dataset 'ZuCo' \
--data_dir "/data/datasets/ZuCo/ZuCo-NR.npy" \
--sparsity 1 \
--batch_size 1 \
--num_epochs 100 \
--frequency 128 \
--num_kernels 8 \
--window_size 16 \
--D 104 \
--p1 1 \
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
