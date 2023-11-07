#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=local-1be95a8bdeaf0e5e73d1b448429ff93821d77b3d
#export WANDB_MODE=offline

cd ../..
python main.py \
--wandb_entity cwg \
--project C42B \
\
--model "FBNetGen" \
--num_repeat 5 \
--dataset "C42B" \
--data_dir "/data/datasets/C42B/C42B128.npy" \
--batch_size 32 \
--num_epochs 200 \
--drop_last True \
\
--mix_up \
--do_train \
--learning_rate 1e-3 \
--schedule 'cos' \
--do_evaluate \
--do_test