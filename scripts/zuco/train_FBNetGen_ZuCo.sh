#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cd ../..
python main.py \
\
--model "FBNetGen" \
--num_repeat 5 \
--wandb_entity cwg \
--project ZuCo-TSR \
\
--dataset 'ZuCo' \
--data_dir "/data/datasets/ZuCo/ZuCo-TSR.npy" \
--batch_size 1 \
--num_epochs 100 \
--drop_last True \
--model_dir "output_dir" \
\
--do_train \
--learning_rate 1e-4 \
--target_learning_rate 1e-5 \
--schedule 'cos' \
--do_evaluate \
--do_test
