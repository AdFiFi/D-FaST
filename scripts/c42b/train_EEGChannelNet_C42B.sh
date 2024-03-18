#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cd ../..
python main.py \
--wandb_entity cwg \
--project C42B \
\
--model "EEGChannelNet" \
--within_subject \
--num_repeat 5 \
--subject_num 9 \
\
--dataset 'C42B' \
--data_dir "/data/datasets/C42B/C42B128.npy" \
--batch_size 32 \
--num_epochs 50 \
--dropout 0.5 \
--drop_last True \
--schedule 'cos' \
--learning_rate 1e-4 \
\
--do_train \
--do_evaluate \
--do_test
