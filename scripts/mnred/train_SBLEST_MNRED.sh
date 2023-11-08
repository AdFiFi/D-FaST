#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="0, 1"

cd ../..
python main.py \
--wandb_entity cwg \
--project BR \
\
--model "SBLEST" \
--num_repeat 5 \
--batch_size 60 \
--num_epochs 2 \
\
--dataset 'MNRED' \
--data_dir "/data/datasets/MNRED/MNRED.npy" \
\
--do_train \
--do_evaluate \
--do_test