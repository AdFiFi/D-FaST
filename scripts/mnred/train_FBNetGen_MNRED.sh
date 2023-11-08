#!/bin/bash
export PYTHONUNBUFFERED=1

cd ../..
python main.py \
\
--model "FBNetGen" \
--num_repeat 5 \
\
--dataset 'MNRED' \
--data_dir "../data/MNRED/MNRED.npy" \
--batch_size 16 \
--num_epochs 100 \
--drop_last True \
--model_dir "output_dir" \
\
--mix_up \
--do_train \
--learning_rate 1e-4 \
--target_learning_rate 1e-5 \
--schedule 'cos' \
--do_evaluate \
--do_test