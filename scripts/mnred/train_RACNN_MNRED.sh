#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

cd ../..
python main.py \
--wandb_entity cwg \
--project MNRED \
\
--model "RACNN" \
--num_repeat 5 \
\
--dataset 'MNRED' \
--data_dir "/data/datasets/MNRED/MNRED.npy" \
--batch_size 8 \
--k 10 \
--num_epochs 100 \
--drop_last True \
--schedule 'cos' \
--learning_rate 1e-4 \
\
--do_train \
--do_evaluate \
--do_test