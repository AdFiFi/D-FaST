#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="0, 1"
#export WANDB_API_KEY=cd4441a5fcdd740b84b45deb6890ecb376bddecb
export WANDB_API_KEY=local-7b0039057e29a88a49ac81830f7cd56f952ec94a
#export WANDB_MODE=offline

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
--dataset 'BR' \
--data_dir "/data/datasets/BR/BR.npy" \
\
--do_train \
--do_evaluate \
--do_test