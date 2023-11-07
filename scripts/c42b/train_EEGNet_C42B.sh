#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=2
export WANDB_API_KEY=local-1be95a8bdeaf0e5e73d1b448429ff93821d77b3d


cd ../..
python main.py \
--wandb_entity cwg \
--project C42B \
\
--model "EEGNet" \
--num_repeat 5 \
\
--dataset "C42B" \
--data_dir "/data/datasets/C42B/C42B128.npy" \
--batch_size 32 \
--num_epochs 200 \
--frequency 128 \
--D 2 \
--num_kernels 8 \
--p1 4 \
--p2 8 \
--dropout 0.5 \
--drop_last True \
--model_dir "output_dir" \
--schedule "cos" \
--learning_rate 1e-4 \
\
--do_train \
--do_evaluate \
--do_test