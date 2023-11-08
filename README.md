# D-FaST: Cognitive Signal Decoding with Disentangled Frequency-Spatial-Temporal Attention

[![stars - D-FaST](https://img.shields.io/github/stars/AdFiFi/D-FaST?style=social)](https://github.com/AdFiFi/D-FaST)
[![forks - D-FaST](https://img.shields.io/github/forks/AdFiFi/D-FaST?style=social)](https://github.com/AdFiFi/D-FaST)
![language](https://img.shields.io/github/languages/top/AdFiFi/D-FaST?color=lightgrey)
![license](https://img.shields.io/github/license/AdFiFi/D-FaST)
---

![D-FaST.jpg](https://github.com/AdFiFi/D-FaST/blob/main/pictures/D-FaST.jpg)

## Dataset

Download the BCIC IV-2A and IV-2B dataset from [here](https://www.bbci.de/competition/iv/index.html).

Download the ZuCo-TSR dataset from [here](https://osf.io/q3zws/).

MNRED dataset will be released in the near future.

## Preprocessing data

Each dataset corresponds to a dataloader and a preprocessing scripts. 
For example, ```smr_preprocess()``` in ```data/smr.py``` process BCIC IV-2A to ```SMR128.npy``` 

## Training

### Default scripts
Use default scripts in ```scripts/``` to train any implemented model in ```model/```. 
All default hyperparameters among these models are tuned for MNRED datasets.

Wandb is needed if visualization of training parameters is wanted

### Customized execution

run script like this:
```bash
python main.py \
--model "DFaST" \
--num_repeat 5 \
--dataset 'MNRED' \
--data_dir "/data/MNRED/MNRED.npy" \
--sparsity 0.6 \
--batch_size 16 \
--num_epochs 100 \
--frequency 200 \
--num_kernels 64 \
--window_size 30 \
--D 30 \
--p1 8 \
--p2 16 \
--drop_last True \
--num_heads 4 \
--learning_rate 1e-4 \
--dropout 0.1 \
--schedule 'cos' \
--do_train \
--do_evaluate
```
For other baseline models, more hyperparameter can be specified in ```config.py``` 
and their own ModelConfig in corresponding model files

## Dependencies
- python==3.10
- braindecode==0.4.85
- einops
- mne
- nilearn==0.9.2
- ptwt==0.1.7
- scikit-learn==1.2.1
- scipy
- torch==2.1.0
- wandb

## Citation

## Contact

Please contact us at ```chenweiguo@nudt.edu.cn```