#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=3
export WANDB_API_KEY=cd4441a5fcdd740b84b45deb6890ecb376bddecb
export WANDB_MODE=offline

cd ../..
python main.py \
\
--model "ShallowConvNet" \
--num_repeat 5 \
\
--dataset 'BR' \
--data_dir "../data/BR/BR.npy" \
--batch_size 16 \
--num_epochs 100 \
--num_kernels 40 \
--drop_last True \
--model_dir "output_dir" \
--schedule 'cos' \
--learning_rate 1e-4 \
\
--do_train \
--do_evaluate \
--do_test