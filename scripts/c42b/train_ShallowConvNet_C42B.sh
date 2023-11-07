#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=cd4441a5fcdd740b84b45deb6890ecb376bddecb
export WANDB_MODE=offline

cd ../..
python main.py \
\
--model "ShallowConvNet" \
--within_subject \
--num_repeat 5 \
--subject_num 9 \
\
--dataset "C42B" \
--data_dir "../data/C42B/C42B128.npy" \
--batch_size 32 \
--num_epochs 200 \
--num_kernels 40 \
--drop_last True \
--model_dir "output_dir" \
--schedule 'cos' \
--learning_rate 1e-3 \
\
--do_train \
--do_evaluate \
--do_test