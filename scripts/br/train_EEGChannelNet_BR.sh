#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
#export WANDB_API_KEY=cd4441a5fcdd740b84b45deb6890ecb376bddecb
export WANDB_API_KEY=local-961e0d861bcf53caabe769f4ce59fe7a1ff8eede
#export WANDB_MODE=offline

cd ../..
python main.py \
--wandb_entity cwg \
--project BR \
\
--model "EEGChannelNet" \
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