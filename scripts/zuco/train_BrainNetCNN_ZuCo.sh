#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1


cd ../..
python main.py \
\
--model "BrainNetCNN" \
--num_repeat 5 \
--wandb_entity cwg \
--project ZuCo-TSR \
\
--dataset 'ZuCo' \
--data_dir "/data/datasets/ZuCo/ZuCo-TSR.npy" \
--batch_size 1 \
--num_epochs 100 \
--drop_last True \
--schedule "cos" \
--learning_rate 1e-4 \
\
--do_train \
--do_evaluate \
--do_test
