#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=cd4441a5fcdd740b84b45deb6890ecb376bddecb
export WANDB_MODE=offline

cd ../..
python main.py \
\
--model "BrainCube10" \
--num_repeat 10 \
\
--dataset 'SMR' \
--data_dir "../data/SMR/SMR.npy" \
--sparsity 0.6 \
--batch_size 16 \
--num_epochs 100 \
--frequency 200 \
--num_kernels 64 \
--window_size 16 \
--D 30 \
--p1 8 \
--p2 16 \
--drop_last False \
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
