import os
from random import shuffle, randrange
import mne
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from .data_config import DataConfig
from .dataset import BaseDataset
from .preprocess import *


class SMRDataset(BaseDataset):
    def __init__(self, data_config: DataConfig, k=0, train=True, subject_id=0, one_hot=True):
        super(SMRDataset, self).__init__(data_config, k, train, subject_id=subject_id, one_hot=one_hot)

    def load_data(self, one_hot=True):
        data = np.load(self.data_config.data_dir, allow_pickle=True).item()
        # time_series = data["timeseries"]
        time_series = data["timeseries"][:, :, :550]
        correlation = data["corr"]
        labels = data["labels"]
        subject_id = data["subject_id"]
        tags = data['tags']

        self.data_config.node_size = self.data_config.node_feature_size = time_series[0].shape[0]
        self.data_config.time_series_size = time_series[0].shape[1]
        self.data_config.num_class = 4

        self.data_config.class_weight = [1, 1, 1, 1]
        self.all_data['time_series'] = time_series
        self.all_data['correlation'] = correlation
        self.all_data['labels'] = labels
        self.all_data['subject_id'] = subject_id
        self.all_data['tags'] = tags
        if self.subject_id:
            self.select_subject()
        groups = np.array([f"{int(s)}_{int(l)}" for s, l in zip(self.all_data['subject_id'], labels)])
        self.train_index, self.test_index = list(self.k_fold.split(self.all_data['time_series'], groups))[self.k]
        # index = np.arange(self.all_data['tags'].shape[0])
        # self.train_index = index[self.all_data['tags'] == 1]
        # self.test_index = index[self.all_data['tags'] == 0]
        self.all_data['labels'] = F.one_hot(torch.from_numpy(self.all_data['labels'] - 1).to(torch.int64)).numpy()
        shuffle(self.train_index)

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
        self.selected = [self.subject_id]
        index = np.sum(self.all_data["subject_id"] == i for i in self.selected) == 1
        self.all_data['time_series'] = self.all_data['time_series'][index]
        self.all_data['correlation'] = self.all_data['correlation'][index]
        self.all_data['labels'] = self.all_data['labels'][index]
        self.all_data['subject_id'] = self.all_data['subject_id'][index]
        self.all_data['tags'] = self.all_data['tags'][index]


def smr_preprocess(path="../data/SMR/"):
    event_id = {'769': 1, '770': 2, '771': 3, '772': 4}
    time_series = pearson = labels = subject_ids = tags = None
    for subject_id in range(1, 10):
        raw = mne.io.read_raw_gdf(os.path.join(path, f"A0{subject_id}T.gdf"), preload=True)
        raw.set_channel_types({"EOG-left": "eog", "EOG-central": "eog", "EOG-right": "eog"})
        raw = raw.filter(4, 38, method='iir').resample(sfreq=128)
        label = loadmat(os.path.join(path, f"A0{subject_id}T.mat"))['classlabel']
        label = label.reshape(label.shape[0])
        events = mne.events_from_annotations(raw, event_id=event_id)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        # epochs = mne.Epochs(raw, picks=picks, events=events[0], event_id=events[1], tmin=0.5, tmax=2.5, baseline=None,
        epochs = mne.Epochs(raw, picks=picks, events=events[0], event_id=events[1], tmin=-0.5, tmax=4, baseline=None,
                            preload=True, verbose=False)
        data = epochs.get_data()
        corr = np.array([np.corrcoef(t) for t in data])

        time_series = data if time_series is None else np.append(time_series, data, axis=0)
        pearson = corr if pearson is None else np.append(pearson, corr, axis=0)
        labels = label if labels is None else np.append(labels, label, axis=0)
        subject_ids = np.ones(label.shape[0]) * subject_id if subject_ids is None \
            else np.append(subject_ids, np.ones(label.shape[0]) * subject_id, axis=0)
    tags = np.ones(labels.shape[0])
    event_id = {'783': 1}
    for subject_id in range(1, 10):
        raw = mne.io.read_raw_gdf(os.path.join(path, f"A0{subject_id}E.gdf"), preload=True)
        raw.set_channel_types({"EOG-left": "eog", "EOG-central": "eog", "EOG-right": "eog"})
        raw = raw.filter(4, 38, method='iir').resample(sfreq=128)
        label = loadmat(os.path.join(path, f"A0{subject_id}E.mat"))['classlabel']
        label = label.reshape(label.shape[0])
        events = mne.events_from_annotations(raw, event_id=event_id)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        # epochs = mne.Epochs(raw, picks=picks, events=events[0], event_id=events[1], tmin=0.5, tmax=2.5, baseline=None,
        epochs = mne.Epochs(raw, picks=picks, events=events[0], event_id=events[1], tmin=-0.5, tmax=4, baseline=None,
                            preload=True, verbose=False)
        data = epochs.get_data()
        corr = np.array([np.corrcoef(t) for t in data])

        time_series = data if time_series is None else np.append(time_series, data, axis=0)
        pearson = corr if pearson is None else np.append(pearson, corr, axis=0)
        labels = label if labels is None else np.append(labels, label, axis=0)
        subject_ids = np.ones(label.shape[0]) * subject_id if subject_ids is None \
            else np.append(subject_ids, np.ones(label.shape[0]) * subject_id, axis=0)
    time_series = data_norm(time_series)
    time_series = preprocess_ea(time_series)
    tags = np.append(tags, np.zeros(labels.shape[0] - tags.shape[0]), axis=0)
    np.save(os.path.join(path, f"SMR128.npy"), {"timeseries": time_series,
                                                 "corr": pearson,
                                                 "labels": labels,
                                                 "subject_id": subject_ids,
                                                 "tags": tags})


if __name__ == '__main__':
    smr_preprocess("../data/SMR")
