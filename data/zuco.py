import os

import mne
import numpy as np
import pandas as pd
import scipy.io as io
import torch
import torch.nn.functional as F
from einops import rearrange

from .data_config import DataConfig
from .dataset import BaseDataset

names = ["ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZJS", "ZKB", "ZKH", "ZKW", "ZMG", "ZPH"]


class ZuCoDataset(BaseDataset):
    def __init__(self, data_config: DataConfig, k=0, train=True, subject_id=0, one_hot=True):
        super(ZuCoDataset, self).__init__(data_config, k, train, subject_id=subject_id, one_hot=one_hot)

    def load_data(self, one_hot=True):
        self.all_data = np.load(self.data_config.data_dir, allow_pickle=True).item()

        self.data_config.node_size = self.data_config.node_feature_size = self.all_data["time_series"][0].shape[0]
        self.padding_and_truncating()

        if "TSR" in self.data_config.data_dir:
            label_map = {k: v for v, k in enumerate(list(set(self.all_data['labels'])))}
            self.all_data['labels'] = np.array([label_map[l] for l in self.all_data['labels']])
            self.data_config.num_class = 10
        elif "SR" in self.data_config.data_dir:
            self.data_config.num_class = 3
            self.all_data['labels'] = self.all_data['labels'] + 1
        else:
            self.all_data['labels'] = (self.all_data['labels'] != "NO-RELATION") * 1
            self.data_config.num_class = 2

        self.data_config.class_weight = [1] * self.data_config.num_class

        if self.subject_id:
            self.select_subject()
        groups = np.array([f"{int(s)}_{int(l)}" for s, l in zip(self.all_data['subject_id'], self.all_data["labels"])])
        self.train_index, self.test_index = list(self.k_fold.split(self.all_data['sentences_time_series'], groups))[self.k]
        if one_hot:
            self.all_data['labels'] = F.one_hot(torch.from_numpy(self.all_data['labels']).to(torch.int64)).numpy()

    def __getitem__(self, item):
        idx = self.train_index if self.train else self.test_index
        sentences_time_series = torch.from_numpy(self.all_data['sentences_time_series'][idx[item]]).float()
        words_time_series = torch.from_numpy(self.all_data['words_time_series'][idx[item]]).float()
        labels = torch.from_numpy(self.all_data['labels'][idx[item]]).to(torch.int64)
        correlation = self.connectivity(sentences_time_series, activate=False)
        words_time_series = rearrange(words_time_series, "c f n w -> n (c f w)")

        # words_time_series = self.norm(words_time_series)
        sentences_time_series = self.norm(sentences_time_series)
        # correlation = self.correlation(time_series)
        # correlation = torch.from_numpy(self.all_data['correlation'][idx[item]]).float()

        return {'time_series': sentences_time_series,
                'correlation': correlation,
                'labels': labels}

    def padding_and_truncating(self):
        all_data = {}
        w_lens = np.array([s.shape[-1] for s in self.all_data["words_time_series"]])
        # s_lens = np.array([s.shape[-1] for s in self.all_data["sentences_time_series"]])
        idx = np.logical_and(w_lens >= 10, w_lens <= 20)
        # idx = np.logical_and(s_lens >= 300, s_lens <= 1000)
        all_data['labels'] = self.all_data['labels'][idx]
        all_data['subject_id'] = self.all_data['subject_id'][idx]
        all_data['mean_time_series'] = self.all_data['mean_time_series'][idx]
        sentences = []
        words_time_series = []
        sentences_time_series = []
        for i, j in enumerate(idx):
            if j:
                sentences.append(self.all_data['sentences'][i])
                sentences_time_series.append(self.all_data['sentences_time_series'][i])
                wts = self.all_data['words_time_series'][i]
                if wts.shape[-1] >= 20:
                    wts = wts[:, :, :, :20]
                else:
                    wts = np.concatenate([wts, np.zeros((3, 8, 104, 20-wts.shape[-1]))], axis=-1)
                words_time_series.append(wts)
        all_data['sentences'] = np.array(sentences)
        all_data['words_time_series'] = np.stack(words_time_series, axis=0)
        # # max_s_lens = np.max(np.array([s.shape[-1] for s in self.all_data["sentences_time_series"]]))
        max_s_lens = 2000
        # for i, sts in enumerate(sentences_time_series):
        #     if sts.shape[-1] >= max_s_lens:
        #         sts = sts[:, -max_s_lens:]
        #     else:
        #         sts = np.concatenate([np.zeros((104, max_s_lens-sts.shape[-1])), sts], axis=-1)
        #     sentences_time_series[i] = sts
        # all_data['sentences_time_series'] = np.stack(sentences_time_series, axis=0)
        all_data['sentences_time_series'] = sentences_time_series
        self.all_data = all_data
        self.data_config.time_series_size = max_s_lens

    def select_subject(self):
        self.selected = [self.subject_id]
        index = np.sum(self.all_data["subject_id"] == i for i in self.selected) == 1
        self.all_data['words_time_series'] = self.all_data['words_time_series'][index]
        # self.all_data['sentences_time_series'] = self.all_data['sentences_time_series'][index]
        self.all_data['sentences_time_series'] = [self.all_data['sentences_time_series'][i]
                                                  for i, j in enumerate(index) if j]
        self.all_data['labels'] = self.all_data['labels'][index]
        self.all_data['subject_id'] = self.all_data['subject_id'][index]
        self.all_data['sentences'] = self.all_data['sentences'][index]


def zuco_preprocess_TSR(path=r"D:\data\ZuCo"):
    subject_ids = []
    labels = []
    sentences_time_series = []
    mean_time_series = []
    words_time_series = []
    sentences = []
    label_path = os.path.join(path, "osfstorage-archive", "task_materials", "relations_labels_task3.csv")
    label_data = pd.read_csv(label_path, sep=";")
    sentence_list = label_data.sentence.tolist()
    labels_list = label_data["relation-type"].tolist()
    channels = [f"eeg{i}" for i in range(104)]
    channel_types = ["eeg"] * 104
    info = mne.create_info(ch_names=channels,
                           sfreq=500,
                           ch_types=channel_types)

    for subject_id, name in enumerate(names):
        file_path = os.path.join(path, "osfstorage-archive", "task3-TSR/Matlab files", f"results{name}_TSR.mat")
        data = io.loadmat(file_path, squeeze_me=True, struct_as_record=False)['sentenceData']

        for i in range(len(data)):
            flag = True
            if isinstance(data[i].rawData, float) or data[i].rawData.shape[-1] < 500:
                continue
            ts = data[i].rawData[:104]
            ts[np.isnan(ts)] = 0

            raw = mne.io.RawArray(ts, info)
            raw.resample(128)
            raw.filter(4, 38)
            ts = raw.get_data()

            freq = ["t1", "t2", "a1", "a2", "g1", "g2", "b1", "b2"]
            mts = []
            for f in freq:
                d_f = eval(f"data[i].mean_{f}")[:104]
                d_f[np.isnan(d_f)] = 0
                mts.append(d_f)
            if len(mts) == 0:
                continue
            mts = np.stack(mts, axis=0)

            word = data[i].word
            ffd = []
            for f in freq:
                d_f = [eval(f"w.FFD_{f}")[:104] for w in word if w.nFixations > 0]
                if len(d_f) == 0:
                    flag = False
                    break
                d_f = np.stack(d_f, axis=-1)
                d_f[np.isnan(d_f)] = 0
                ffd.append(d_f)
            if not flag:
                continue
            ffd = np.stack(ffd, axis=0)
            trt = []
            for f in freq:
                d_f = [eval(f"w.TRT_{f}")[:104] for w in word if w.nFixations > 0]
                d_f = np.stack(d_f, axis=-1)
                d_f[np.isnan(d_f)] = 0
                trt.append(d_f)
            trt = np.stack(trt, axis=0)
            gd = []
            for f in freq:
                d_f = [eval(f"w.GD_{f}")[:104] for w in word if w.nFixations > 0]
                d_f = np.stack(d_f, axis=-1)
                d_f[np.isnan(d_f)] = 0
                gd.append(d_f)
            gd = np.stack(gd, axis=0)

            words_time_series.append(np.stack([ffd, trt, gd], axis=0))
            sentences_time_series.append(ts)
            mean_time_series.append(mts)
            sentence = data[i].content
            sentence = sentence.replace("(40 km�)", "(40 km)")
            sentences.append(sentence)

            sentence_idx = sentence_list.index(sentence)
            label = labels_list[sentence_idx]
            labels.append(label)
            subject_ids.append(subject_id + 1)

    mean_time_series = np.stack(mean_time_series, axis=0)
    labels = np.array(labels)
    subject_ids = np.array(subject_ids)
    np.save(os.path.join(path, f"ZuCo-TSR.npy"), {"labels": labels,
                                                  "sentences": sentences,
                                                  "words_time_series": words_time_series,
                                                  "mean_time_series": mean_time_series,
                                                  "sentences_time_series": sentences_time_series,
                                                  "subject_id": subject_ids})
