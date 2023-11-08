from abc import abstractmethod

import torch
from nilearn import connectome
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from .data_config import DataConfig


class BaseDataset(Dataset):
    def __init__(self, data_config: DataConfig, k=0, train=True, subject_id=0, one_hot=True):
        super(BaseDataset, self).__init__()
        self.data_config = data_config
        self.train = train
        if data_config.n_splits-1:
            self.k_fold = StratifiedKFold(n_splits=data_config.n_splits, shuffle=True, random_state=42)
        else:
            self.k_fold = None
        self.k = k
        self.subject_id = subject_id
        self.selected = []
        self.all_data = {}
        self.train_index = None
        self.test_index = None
        self.train_data = None
        self.test_data = None
        self.load_data(one_hot=one_hot)

    @abstractmethod
    def load_data(self, one_hot=True):
        pass

    @staticmethod
    def connectivity(time_series, activate=True):
        conn_measure = connectome.ConnectivityMeasure(kind='correlation')
        # conn_measure = connectome.ConnectivityMeasure(kind='correlation', cov_estimator=OAS(store_precision=False))
        connectivity = conn_measure.fit_transform(time_series.T.unsqueeze(0).numpy())[0]
        connectivity = torch.from_numpy(connectivity)
        if activate:
            connectivity = torch.arctanh(connectivity)
            connectivity = torch.clamp(connectivity, -1.0, 1.0)
            diag = torch.diag_embed(torch.diag(connectivity))
            connectivity = connectivity - diag
        return connectivity

    @staticmethod
    def correlation(time_series, activate=True):
        feature = torch.einsum('nt, mt ->nm', time_series, time_series) / (time_series.size(1)-1)
        feature = torch.clamp(feature, -1.0, 1.0)
        if activate:
            feature = torch.arctanh(feature)
            feature = torch.clamp(feature, -1.0, 1.0)
            diag = torch.diag_embed(torch.diag(feature))
            feature = feature - diag
        return feature

    @staticmethod
    def norm(time_series):
        time_series -= torch.mean(time_series, dim=1, keepdim=True)
        std = torch.std(time_series, dim=1, keepdim=True)
        std[std < torch.finfo(torch.float64).eps] = 1.
        time_series /= std
        return time_series

    def __len__(self):
        return len(self.train_index) if self.train else len(self.test_index)

    @abstractmethod
    def __getitem__(self, item):
        pass
