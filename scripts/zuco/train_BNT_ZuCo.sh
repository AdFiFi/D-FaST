#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cd ../..
python main.py \
--num_repeat 5 \
--wandb_entity cwg \
--project ZuCo-NR \
\
--model "BNT" \
\
--dataset 'ZuCo' \
--data_dir "/data/datasets/ZuCo/ZuCo-NR.npy" \
--batch_size 1 \
--num_epochs 100 \
--drop_last True \
--model_dir "output_dir" \
--schedule 'cos' \
--learning_rate 1e-4 \
\
--do_train \
--do_evaluate \
--do_test
