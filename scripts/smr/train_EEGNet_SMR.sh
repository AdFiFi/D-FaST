#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=local-1be95a8bdeaf0e5e73d1b448429ff93821d77b3d


cd ../..
python main.py \
--wandb_entity cwg \
--project SMR \
\
--model "EEGNet" \
--num_repeat 5 \
\
--dataset "SMR" \
--data_dir "/data/datasets/SMR/SMR128.npy" \
--batch_size 32 \
--num_epochs 200 \
--frequency 128 \
--D 22 \
--num_kernels 16 \
--p1 4 \
--p2 8 \
--dropout 0.5 \
--drop_last True \
--model_dir "output_dir" \
--schedule "cos" \
--learning_rate 1e-3 \
\
--do_train \
--do_evaluate \
--do_test