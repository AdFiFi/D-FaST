#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=2
export WANDB_API_KEY=cd4441a5fcdd740b84b45deb6890ecb376bddecb
export WANDB_MODE=offline

cd ../..
python main.py \
\
--model "BrainCube7" \
--num_repeat 5 \
\
--dataset 'SMR' \
--data_dir "../data/SMR/SMR.npy" \
--percentage 1. \
--sparsity 1 \
--batch_size 64 \
--num_epochs 50 \
--drop_last False \
--num_heads 2 \
--d_model 128 \
--window_size 50 \
--window_strid 3 \
--dim_feedforward 1024 \
--num_node_temporal_layers 2 \
--num_layers 2 \
--num_graph_temporal_layers 2 \
--attention_depth 1 \
--learning_rate 1e-4 \
--dropout 0.3 \
--schedule 'cos' \
--mix_up \
\
--do_train \
--do_evaluate \
--do_test
