import os
from scipy.io import loadmat
import h5py

from random import shuffle, randrange

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

from .data_config import DataConfig
from .dataset import BaseDataset


class MNREDDataset(BaseDataset):
    def __init__(self, data_config: DataConfig, k=0, train=True, subject_id=0, one_hot=True):
        super(MNREDDataset, self).__init__(data_config, k, train, subject_id=subject_id, one_hot=one_hot)

    def load_data(self, one_hot=True):
        data = np.load(self.data_config.data_dir, allow_pickle=True).item()
        time_series = data["timeseries"]
        correlation = data["corr"]
        labels = data["labels"]
        subject_id = data["subject_id"]

        self.data_config.node_size = self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.num_class = 2

        self.data_config.class_weight = [1, 2]
        self.all_data['time_series'] = time_series
        self.all_data['correlation'] = correlation
        self.all_data['labels'] = labels
        self.all_data['subject_id'] = subject_id

        self.select_subject()
        groups = np.array([f"{int(s)}_{int(l)}" for s, l in zip(self.all_data['subject_id'], labels)])
        self.train_index, self.test_index = list(self.k_fold.split(self.all_data['time_series'], groups))[self.k]
        if one_hot:
            self.all_data['labels'] = F.one_hot(torch.from_numpy(self.all_data['labels']).to(torch.int64)).numpy()

    def __getitem__(self, item):
        idx = self.train_index if self.train else self.test_index
        time_series = torch.from_numpy(self.all_data['time_series'][idx[item]]).float()
        labels = torch.from_numpy(self.all_data['labels'][idx[item]]).to(torch.int64)

        sampling_init = (randrange(time_series.size(-1) - self.data_config.time_series_size)) \
            if self.data_config.dynamic else 0
        time_series = time_series[:, sampling_init:sampling_init + self.data_config.time_series_size]
        correlation = self.connectivity(time_series, activate=False)
        # time_series = self.norm(time_series)
        # correlation = torch.from_numpy(self.all_data['correlation'][idx[item]]).float()

        return {'time_series': time_series,
                'correlation': correlation,
                'labels': labels}

    def select_subject(self):
        self.selected = [self.subject_id]
        index = np.sum(self.all_data["subject_id"] == i for i in self.selected) == 1
        self.all_data['time_series'] = self.all_data['time_series'][index]
        self.all_data['correlation'] = self.all_data['correlation'][index]
        self.all_data['labels'] = self.all_data['labels'][index]
        self.all_data['subject_id'] = self.all_data['subject_id'][index]


def eeg_preprocess_test(path):
    all_data = loadmat(os.path.join(path, "Data0324.mat"))
    time_series = all_data['data']
    pearson = np.array([np.corrcoef(t) for t in time_series])
    labels = all_data['label'][0]
    subject_id = all_data['subject'][0]
    np.save(os.path.join(path, "EEG.npy"), {"timeseries": time_series,
                                            "corr": pearson,
                                            "labels": labels,
                                            "subject_id": subject_id})


if __name__ == '__main__':
    eeg_preprocess_test("../data/EEG")
