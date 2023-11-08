#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=2

cd ../..
python main.py \
\
--model "EEGNet" \
--num_repeat 5 \
\
--dataset 'MNRED' \
--data_dir "../data/MNRED/MNRED.npy" \
--batch_size 16 \
--num_epochs 200 \
--frequency 64 \
--D 5 \
--num_kernels 64 \
--p1 8 \
--p2 16 \
--dropout 0.5 \
--drop_last True \
--mix_up \
--model_dir "output_dir" \
--schedule 'cos' \
--learning_rate 1e-4 \
\
--do_train \
--do_evaluate \
--do_test