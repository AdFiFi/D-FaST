#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=4
#export WANDB_API_KEY=cd4441a5fcdd740b84b45deb6890ecb376bddecb
export WANDB_API_KEY=local-7b0039057e29a88a49ac81830f7cd56f952ec94a
#export WANDB_MODE=offline

cd ../..
python main.py \
--wandb_entity cwg \
--project BR3 \
\
--model "TCACNet" \
--num_repeat 5 \
\
--dataset 'BR' \
--data_dir "/data/datasets/BR/BR.npy" \
--batch_size 16 \
--num_epochs 100 \
--drop_last True \
--schedule 'cos' \
--learning_rate 1e-4 \
\
--do_train \
--do_evaluate \
--do_test