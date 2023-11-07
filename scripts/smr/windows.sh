#!/bin/bash
export PYTHONUNBUFFERED=1


cd ../..
#export CUDA_VISIBLE_DEVICES=0
#python main.py --project "FaST-P-SMR2" --model "FaSTP" --window_size 2 --num_repeat 5 --dataset "SMR" --data_dir "../data/SMR/SMR128.npy" --sparsity 0.6 --batch_size 32 --num_epochs 200 --frequency 128 --num_kernels 64 --D 8 --p1 4 --p2 8 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &

export CUDA_VISIBLE_DEVICES=1
python main.py --project "FaST-P-SMR2" --model "FaSTP" --mix_up --window_size 4 --num_repeat 5 --dataset "SMR" --data_dir "../data/SMR/SMR128.npy" --sparsity 0.6 --batch_size 32 --num_epochs 200 --frequency 128 --num_kernels 64 --D 8 --p1 4 --p2 8 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &

#export CUDA_VISIBLE_DEVICES=2
#python main.py --project "FaST-P-SMR2" --model "FaSTP" --window_size 8 --num_repeat 5 --dataset "SMR" --data_dir "../data/SMR/SMR128.npy" --sparsity 0.6 --batch_size 32 --num_epochs 200 --frequency 128 --num_kernels 64 --D 8 --p1 4 --p2 8 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &
#
#export CUDA_VISIBLE_DEVICES=3
#python main.py --project "FaST-P-SMR2" --model "FaSTP" --window_size 16 --num_repeat 5 --dataset "SMR" --data_dir "../data/SMR/SMR128.npy" --sparsity 0.6 --batch_size 32 --num_epochs 200 --frequency 128 --num_kernels 64 --D 8 --p1 4 --p2 8 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &
#
#export CUDA_VISIBLE_DEVICES=0
#python main.py --project "FaST-P-SMR2" --model "FaSTP" --window_size 32 --num_repeat 5 --dataset "SMR" --data_dir "../data/SMR/SMR128.npy" --sparsity 0.6 --batch_size 32 --num_epochs 200 --frequency 128 --num_kernels 64 --D 8 --p1 4 --p2 8 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &
#
#export CUDA_VISIBLE_DEVICES=1
#python main.py --project "FaST-P-SMR2" --model "FaSTP" --window_size 64 --num_repeat 5 --dataset "SMR" --data_dir "../data/SMR/SMR128.npy" --sparsity 0.6 --batch_size 32 --num_epochs 200 --frequency 128 --num_kernels 64 --D 8 --p1 4 --p2 8 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &
#
#export CUDA_VISIBLE_DEVICES=2
#python main.py --project "FaST-P-SMR2" --model "FaSTP" --window_size 128 --num_repeat 5 --dataset "SMR" --data_dir "../data/SMR/SMR128.npy" --sparsity 0.6 --batch_size 32 --num_epochs 200 --frequency 128 --num_kernels 64 --D 8 --p1 4 --p2 8 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &
