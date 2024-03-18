# D-FaST: Cognitive Signal Decoding with Disentangled Frequency-Spatial-Temporal Attention

[![stars - D-FaST](https://img.shields.io/github/stars/AdFiFi/D-FaST?style=social)](https://github.com/AdFiFi/D-FaST)
[![forks - D-FaST](https://img.shields.io/github/forks/AdFiFi/D-FaST?style=social)](https://github.com/AdFiFi/D-FaST)
![language](https://img.shields.io/github/languages/top/AdFiFi/D-FaST?color=lightgrey)
![license](https://img.shields.io/github/license/AdFiFi/D-FaST)
---

## Approach

![D-FaST.jpg](pictures/D-FaST.jpg)

## Dataset

Download the BCIC IV-2A and IV-2B dataset from [here](https://www.bbci.de/competition/iv/index.html).

Download the ZuCo-TSR dataset from [here](https://osf.io/q3zws/).

MNRED dataset will be released in the near future.

## Preprocessing Data

Each dataset corresponds to a dataloader and a preprocessing scripts. 
For example, ```smr_preprocess()``` in ```data/smr.py``` process BCIC IV-2A to ```SMR128.npy``` 

## Training

### Default Scripts
Use default scripts in ```scripts/``` to train any implemented model in ```model/```. 
All default hyperparameters among these models are tuned for MNRED datasets.

Wandb is needed if visualization of training parameters is wanted

### Customized Execution

run script like this:
```bash
python main.py \
--model DFaST \
--num_repeat 5 \
--dataset MNRED \
--data_dir /data/MNRED/MNRED.npy \
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
--schedule cos \
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

```
@ARTICLE{10445385,
  author={Chen, WeiGuo and Wang, Changjian and Xu, Kele and Yuan, Yuan and Bai, Yanru and Zhang, Dongsong},
  journal={IEEE Transactions on Cognitive and Developmental Systems}, 
  title={D-FaST: Cognitive Signal Decoding with Disentangled Frequency-Spatial-Temporal Attention}, 
  year={2024},
  volume={},
  number={},
  pages={1-18},
  keywords={Feature extraction;Frequency-domain analysis;Decoding;Electroencephalography;Data mining;Brain modeling;Time series analysis;Cognitive Language Processing (CLP);Cognitive Signal Decoding (CSD);Frequency-spatial-temporal domain attention},
  doi={10.1109/TCDS.2024.3370261}}

```

## Contact Us

If you are interested to leave a message, please feel free to send any email to us at ```chenweiguo@nudt.edu.cn```
