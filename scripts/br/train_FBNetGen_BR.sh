#!/bin/bash
export PYTHONUNBUFFERED=1
export WANDB_API_KEY=cd4441a5fcdd740b84b45deb6890ecb376bddecb
export WANDB_MODE=offline

cd ../..
python main.py \
\
--model "FBNetGen" \
--num_repeat 5 \
\
--dataset 'BR' \
--data_dir "../data/BR/BR.npy" \
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