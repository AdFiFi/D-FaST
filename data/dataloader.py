import torch
import copy
import torch.utils.data as utils
from .data_config import DataConfig
from typing import Dict
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import numpy as np
import torch.nn.functional as F


def init_StratifiedKFold_dataloader(data_config: DataConfig,
                                    dataset) -> Dict[str, utils.DataLoader]:
    train_dataset = dataset
    # train_weight = np.ones(len(train_dataset)) * 0.3
    # train_weight[train_dataset.all_data['labels'][train_dataset.train_index][:, 1] == 1] = 0.7
    test_dataset = copy.deepcopy(dataset)
    test_dataset.train = False
    # train_sampler = utils.WeightedRandomSampler(weights=train_weight,
    #                                             num_samples=data_config.batch_size,
    #                                             replacement=False)
    train_dataloader = utils.DataLoader(
        train_dataset,
        # sampler=train_sampler,
        batch_size=data_config.batch_size if data_config.dataset != "ZuCo" else 1,
        # batch_size=data_config.batch_size,
        shuffle=True,
        drop_last=data_config.drop_last,
        num_workers=5)

    test_dataloader = utils.DataLoader(
        test_dataset,
        batch_size=data_config.batch_size if data_config.dataset != "ZuCo" else 1,
        # batch_size=data_config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=5)

    return {"train": train_dataloader,
            "test": test_dataloader}


def init_distributed_dataloader(data_config: DataConfig,
                                dataset) -> Dict[str, utils.DataLoader]:
    train_dataset = dataset
    test_dataset = copy.deepcopy(dataset)
    test_dataset.train = False

    train_sampler = utils.distributed.DistributedSampler(train_dataset)
    train_dataloader = utils.DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        drop_last=data_config.drop_last,
        sampler=train_sampler,
        num_workers=5)

    test_dataloader = utils.DataLoader(
        test_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=5)

    return {"train": train_dataloader,
            "test": test_dataloader}
