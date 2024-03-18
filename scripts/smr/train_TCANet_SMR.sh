#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=4

cd ../..
python main.py \
--wandb_entity cwg \
--project SMR \
\
--model "TCANet" \
--within_subject \
--num_repeat 5 \
--subject_num 9 \
\
--dataset SMR \
--data_dir "/data/datasets/SMR/SMR128.npy" \
--batch_size 32 \
--num_epochs 200 \
--drop_last True \
--schedule cos \
--learning_rate 1e-4 \
\
--do_train \
--do_evaluate \
--do_test
