#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
#export WANDB_API_KEY=cd4441a5fcdd740b84b45deb6890ecb376bddecb
export WANDB_API_KEY=local-1be95a8bdeaf0e5e73d1b448429ff93821d77b3d
#export WANDB_MODE=offline

cd ../..
python main.py \
--wandb_entity cwg \
--project C42B1 \
\
--model "SBLEST" \
--num_repeat 5 \
--batch_size 64 \
--num_epochs 2 \
\
--dataset 'C42B' \
--data_dir "/data/datasets/C42B/C42B128.npy" \
\
--do_train \
--do_evaluate \
--do_test