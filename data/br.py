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


class BRDataset(BaseDataset):
    def __init__(self, data_config: DataConfig, k=0, train=True, subject_id=0, one_hot=True):
        super(BRDataset, self).__init__(data_config, k, train, subject_id=subject_id, one_hot=one_hot)

    def load_data(self, one_hot=True):
        data = np.load(self.data_config.data_dir, allow_pickle=True).item()
        time_series = data["timeseries"]
        correlation = data["corr"]
        labels = data["labels"]
        subject_id = data["subject_id"]

        self.data_config.node_size = self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.num_class = 2

        # self.data_config.class_weight = [1, 1]
        self.data_config.class_weight = [1, 2]
        # self.data_config.class_weight = [0.3, 0.7]
        self.all_data['time_series'] = time_series
        self.all_data['correlation'] = correlation
        self.all_data['labels'] = labels
        self.all_data['subject_id'] = subject_id

        # p_index = labels == 1
        # p_time_series = time_series[p_index]
        # p_correlation = correlation[p_index]
        # p_subject_id = subject_id[p_index]
        #
        # n_index = labels == 0
        # n_time_series = time_series[n_index]
        # n_correlation = correlation[n_index]
        # n_subject_id = subject_id[n_index]
        #
        # n_split = StratifiedShuffleSplit(n_splits=1, test_size=4 / 7, train_size=3 / 7)
        # n_use_index, n_drop_index = list(n_split.split(n_time_series, n_subject_id))[0]
        # n_use_time_series = n_time_series[n_use_index]
        # n_use_correlation = n_correlation[n_use_index]
        # n_use_subject_id = n_subject_id[n_use_index]
        #
        # labels = np.append(np.zeros(n_use_index.shape[0], dtype="int64"), np.ones(p_time_series.shape[0], dtype="int64"))
        #
        # self.all_data['time_series'] = np.append(n_use_time_series, p_time_series, axis=0)
        # self.all_data['correlation'] = np.append(n_use_correlation, p_correlation, axis=0)
        # self.all_data['labels'] = labels
        # self.all_data['subject_id'] = np.append(n_use_subject_id, p_subject_id, axis=0)

        self.select_subject()
        groups = np.array([f"{int(s)}_{int(l)}" for s, l in zip(self.all_data['subject_id'], labels)])
        self.train_index, self.test_index = list(self.k_fold.split(self.all_data['time_series'], groups))[self.k]
        # idx = (self.all_data['subject_id'] == self.selected[self.k * 2]) + \
        #       (self.all_data['subject_id'] == self.selected[self.k * 2 + 1])
        # self.test_index = np.where(idx)[0]
        # self.train_index = np.where(~idx)[0]
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
        # correlation = self.correlation(time_series)
        # correlation = torch.from_numpy(self.all_data['correlation'][idx[item]]).float()

        return {'time_series': time_series,
                'correlation': correlation,
                'labels': labels}

    def select_subject(self):
        # if self.data_config.subject_id == 0:
        #     return
        # else:
        #     index = self.all_data["subject_id"] == self.data_config.subject_id
        #     self.all_data['time_series'] = self.all_data['time_series'][index]
        #     self.all_data['correlation'] = self.all_data['correlation'][index]
        #     self.all_data['labels'] = self.all_data['labels'][index]
        #     self.all_data['subject_id'] = self.all_data['subject_id'][index]
        self.selected = [6, 7, 8, 11, 13, 14, 15, 21, 22, 24]
        # self.selected = [31]
        # index = self.all_data["subject_id"] == self.data_config.subject_id
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


def eeg_preprocess_multi_freq(path):
    freq_files = ["Data_alpha.mat", "Data_beta.mat", "Data_gamma.mat", "Data_theta.mat"]
    all_data = loadmat(os.path.join(path, "Data0325.mat"))
    time_series = all_data['data']
    B, N, T = time_series.shape
    pearson = np.array([np.corrcoef(t) for t in time_series]).reshape((B, 1, N, N))
    time_series = time_series.reshape((B, 1, N, T))
    labels = all_data['label'][0]
    subject_id = all_data['subject'][0]
    for file in freq_files:
        ts = np.array(h5py.File(os.path.join(path, file))['data_bp']).T
        time_series = np.append(time_series, ts.reshape((B, 1, N, T)), axis=1)
        pearson = np.append(pearson, np.array([np.corrcoef(t) for t in ts]).reshape((B, 1, N, N)), axis=1)
    np.save(os.path.join(path, "EEG_5.npy"), {"timeseries": time_series,
                                              "corr": pearson,
                                              "labels": labels,
                                              "subject_id": subject_id})


if __name__ == '__main__':
    eeg_preprocess_test("../data/EEG")
