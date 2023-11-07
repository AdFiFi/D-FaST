import numpy as np
import pandas as pd
import torch

from scipy.signal import firwin, lfilter, filtfilt, butter
from scipy.linalg import sqrtm


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean: np.array, std: np.array):
        self.mean = mean
        self.std = std

    def transform(self, data: np.array):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.array):
        return (data * self.std) + self.mean


def continues_mixup_data(*xs, y1=None, y2=None, alpha=1.0, beta=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1
    batch_size = y1.size()[0]
    index = torch.randperm(batch_size)
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    y1 = lam * y1 + (1-lam) * y1[index]
    y2 = lam * y2 + (1-lam) * y2[index] if y2 is not None else y2
    return *new_xs, y1, y2

    # batch_size = y.size()[0]
    # if alpha > 0:
    #     lam = torch.tensor(np.random.beta(alpha, alpha, batch_size)).float()
    # else:
    #     lam = torch.ones(batch_size)
    # index = torch.randperm(batch_size)
    # new_xs = [torch.einsum('b, bnf -> bnf', lam, x) + torch.einsum('b, bnf -> bnf', 1 - lam, x[index, :]) for x in xs]
    # y = torch.einsum('b, bc -> bc', lam, y) + torch.einsum('b, bc -> bc', 1-lam, y[index])
    # return *new_xs, y


def data_norm(data):
    data_copy = np.copy(data)
    for i in range(len(data)):
        data_copy[i] = data_copy[i] / np.max(abs(data[i]))
        # data_copy[i] = exponential_running_standardize(data_copy[i].T).T

    return data_copy


def exponential_running_standardize(
        data, factor_new=0.001, init_block_size=1000, eps=1e-4
):
    """

    """

    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_block_standardized = (
                                          data[0:init_block_size] - init_mean
                                  ) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized


def bandpass_cnt(data, low_cut_hz, high_cut_hz, fs, filt_order=200, zero_phase=False):
    # nyq_freq = 0.5 * fs
    # low = low_cut_hz / nyq_freq
    # high = high_cut_hz / nyq_freq

    # win = firwin(filt_order, [low, high], window='blackman', ass_zero='bandpass')
    win = firwin(filt_order, [low_cut_hz, high_cut_hz], window='blackman', fs=fs, pass_zero='bandpass')

    data_bandpassed = lfilter(win, 1, data)
    if zero_phase:
        data_bandpassed = filtfilt(win, 1, data)
    return data_bandpassed


def preprocess_ea(data):
    R_bar = np.zeros((data.shape[1], data.shape[1]))
    for i in range(len(data)):
        R_bar += np.dot(data[i], data[i].T)
    R_bar_mean = R_bar / len(data)
    # assert (R_bar_mean >= 0 ).all(), 'Before squr,all element must >=0'

    for i in range(len(data)):
        data[i] = np.dot(np.linalg.inv(sqrtm(R_bar_mean)), data[i])
    return data