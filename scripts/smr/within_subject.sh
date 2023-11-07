#!/bin/bash
export PYTHONUNBUFFERED=1
#export WANDB_API_KEY=local-1a6c67774093a56b310d8313b6821f2d98e59678
#export WANDB_MODE=offline

cd ../..
export CUDA_VISIBLE_DEVICES=0
python main.py --model "FaSTP" --project "FaST-P-SMR1" --num_kernels 8 --window_size 3 --data_dir "../data/SMR/SMR128.npy" --within_subject --num_repeat 5 --subject_num 9 --dataset 'SMR' --sparsity 1 --batch_size 32 --num_epochs 200 --frequency 128 --mix_up --D 4 --p1 4 --p2 8 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &

export CUDA_VISIBLE_DEVICES=1
python main.py --model "FaSTP" --project "FaST-P-SMR1" --num_kernels 16 --window_size 3 --data_dir "../data/SMR/SMR128.npy" --within_subject --num_repeat 5 --subject_num 9 --dataset 'SMR' --sparsity 1 --batch_size 32 --num_epochs 200 --frequency 128 --mix_up --D 4 --p1 4 --p2 8 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &

export CUDA_VISIBLE_DEVICES=2
python main.py --model "FaSTP" --project "FaST-P-SMR1" --num_kernels 32 --window_size 3 --data_dir "../data/SMR/SMR128.npy" --within_subject --num_repeat 5 --subject_num 9 --dataset 'SMR' --sparsity 1 --batch_size 32 --num_epochs 200 --frequency 128 --mix_up --D 4 --p1 4 --p2 8 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &

export CUDA_VISIBLE_DEVICES=3
python main.py --model "FaSTP" --project "FaST-P-SMR1" --num_kernels 64 --window_size 3 --data_dir "../data/SMR/SMR128.npy" --within_subject --num_repeat 5 --subject_num 9 --dataset 'SMR' --sparsity 1 --batch_size 32 --num_epochs 200 --frequency 128 --mix_up --D 4 --p1 4 --p2 8 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &

export CUDA_VISIBLE_DEVICES=1
python main.py --model "FaSTP" --project "FaST-P-SMR1" --num_kernels 128 --window_size 3 --data_dir "../data/SMR/SMR128.npy" --within_subject --num_repeat 5 --subject_num 9 --dataset 'SMR' --sparsity 1 --batch_size 32 --num_epochs 200 --frequency 128 --mix_up --D 4 --p1 4 --p2 8 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &

#export CUDA_VISIBLE_DEVICES=1
#python main.py --model "EEGNet" --within_subject --num_repeat 5 --subject_num 9 --dataset "SMR" --data_dir "../data/SMR/SMR128.npy" --batch_size 32 --num_epochs 200 --frequency 128 --D 2 --num_kernels 16 --p1 4 --p2 8 --dropout 0.5 --drop_last True --model_dir "output_dir" --schedule "cos" --learning_rate 1e-3 --do_train --do_evaluate --do_test &
#
#export CUDA_VISIBLE_DEVICES=2
#python main.py --model "EEGNet" --within_subject --num_repeat 5 --subject_num 9 --dataset "SMR" --data_dir "../data/SMR/SMR128.npy" --batch_size 32 --num_epochs 200 --frequency 128 --D 22 --num_kernels 16 --p1 4 --p2 8 --dropout 0.5 --drop_last True --model_dir "output_dir" --schedule "cos" --learning_rate 1e-3 --do_train --do_evaluate --do_test &
#
#export CUDA_VISIBLE_DEVICES=3
#python main.py --model "ShallowConvNet" --within_subject --num_repeat 5 --subject_num 9 --dataset "SMR" --data_dir "../data/SMR/SMR128.npy" --batch_size 32 --num_epochs 200 --num_kernels 40 --drop_last True --model_dir "output_dir" --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test &
#
#export CUDA_VISIBLE_DEVICES=0
#python main.py --model "DeepConvNet" --within_subject --num_repeat 5 --subject_num 9 --dataset "SMR" --data_dir "../data/SMR/SMR128.npy" --batch_size 32 --num_epochs 200 --drop_last True --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test &
#
#export CUDA_VISIBLE_DEVICES=1
#python main.py --model "LMDA" --within_subject --num_repeat 5 --subject_num 9 --dataset "SMR" --data_dir "../data/SMR/SMR128.npy" --batch_size 32 --num_epochs 200 --num_kernels 24 --drop_last True --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test &
#
#export CUDA_VISIBLE_DEVICES=2
#python main.py --model "BrainNetCNN" --within_subject --num_repeat 5 --subject_num 9 --dataset 'SMR' --data_dir "../data/SMR/SMR128.npy" --batch_size 32 --num_epochs 200 --drop_last True --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test &
#
##export CUDA_VISIBLE_DEVICES=1
##python main.py --model "FBNetGen" --within_subject --num_repeat 5 --subject_num 9 --dataset "SMR" --data_dir "../data/SMR/SMR128.npy" --batch_size 32 --num_epochs 200 --drop_last True --mix_up --do_train --learning_rate 1e-3 --schedule 'cos' --do_evaluate --do_test &
#
#export CUDA_VISIBLE_DEVICES=3
#python main.py --model "BNT" --within_subject --num_repeat 5 --subject_num 9 --dataset 'SMR' --data_dir "../data/SMR/SMR128.npy" --batch_size 32 --num_epochs 200 --drop_last True --model_dir "output_dir" --schedule 'cos' --learning_rate 1e-3 --do_train --do_evaluate --do_test &
