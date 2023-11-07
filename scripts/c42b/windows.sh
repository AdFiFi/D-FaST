#!/bin/bash
export PYTHONUNBUFFERED=1


cd ../..
#export CUDA_VISIBLE_DEVICES=0
#python main.py --project "FaST-P-C42B2" --window_size 2 --model "FaSTP" --num_repeat 5 --dataset "C42B" --data_dir "../data/C42B/C42B128.npy" --sparsity 1 --batch_size 32 --num_epochs 200 --frequency 128 --num_kernels 64 --D 1 --p1 16 --p2 1 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &
#
#export CUDA_VISIBLE_DEVICES=1
#python main.py --project "FaST-P-C42B2" --window_size 4 --model "FaSTP" --num_repeat 5 --dataset "C42B" --data_dir "../data/C42B/C42B128.npy" --sparsity 1 --batch_size 32 --num_epochs 200 --frequency 128 --num_kernels 64 --D 1 --p1 16 --p2 1 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &
#
#export CUDA_VISIBLE_DEVICES=2
#python main.py --project "FaST-P-C42B2" --window_size 8 --model "FaSTP" --num_repeat 5 --dataset "C42B" --data_dir "../data/C42B/C42B128.npy" --sparsity 1 --batch_size 32 --num_epochs 200 --frequency 128 --num_kernels 64 --D 1 --p1 16 --p2 1 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &
#
#export CUDA_VISIBLE_DEVICES=3
#python main.py --project "FaST-P-C42B2" --window_size 16 --model "FaSTP" --num_repeat 5 --dataset "C42B" --data_dir "../data/C42B/C42B128.npy" --sparsity 1 --batch_size 32 --num_epochs 200 --frequency 128 --num_kernels 64 --D 1 --p1 16 --p2 1 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &
#
#export CUDA_VISIBLE_DEVICES=0
#python main.py --project "FaST-P-C42B2" --window_size 32 --model "FaSTP" --num_repeat 5 --dataset "C42B" --data_dir "../data/C42B/C42B128.npy" --sparsity 1 --batch_size 32 --num_epochs 200 --frequency 128 --num_kernels 64 --D 1 --p1 16 --p2 1 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &
#
#export CUDA_VISIBLE_DEVICES=1
#python main.py --project "FaST-P-C42B2" --window_size 64 --model "FaSTP" --num_repeat 5 --dataset "C42B" --data_dir "../data/C42B/C42B128.npy" --sparsity 1 --batch_size 32 --num_epochs 200 --frequency 128 --num_kernels 64 --D 1 --p1 16 --p2 1 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &

export CUDA_VISIBLE_DEVICES=2
python main.py --project "FaST-P-C42B2" --window_size 128 --model "FaSTP" --num_repeat 5 --dataset "C42B" --data_dir "../data/C42B/C42B128.npy" --sparsity 1 --batch_size 32 --num_epochs 200 --frequency 128 --num_kernels 64 --D 1 --p1 16 --p2 1 --drop_last True --num_heads 4 --distill --num_layers 1 --learning_rate 1e-3 --dropout 0.5 --schedule 'cos' --do_train --do_evaluate --do_test &
