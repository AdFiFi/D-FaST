import torch

from trainers import DFaSTTrainer
from config import init_config
from data import *


def mva_attention():
    args = init_config()
    trainer = DFaSTTrainer(args, task_id=0)
    trainer.load_model()
    dataloader = trainer.data_loaders['test']
    trainer.model.eval()
    attention = None
    for inputs in dataloader:
        time_series = trainer.prepare_inputs_kwargs(inputs)['time_series'].unsqueeze(1)
        attention = trainer.model.fast_p.mva.get_attention(time_series)
        attention = attention.mean(0).detach().numpy()
        break


def dca_graph():
    args = init_config()
    data_config = DataConfig(args)
    datasets = BRDataset(data_config, 0)
    time_series = datasets.all_data['time_series'][datasets.train_index]
    subject_id = datasets.all_data['subject_id'][datasets.train_index]
    labels = datasets.all_data['labels'][datasets.train_index]
    ind = subject_id == 6
    time_series = time_series[ind]
    labels = labels[ind]
    t0 = time_series[labels[:, 0] == 1].mean(0)
    t1 = time_series[labels[:, 1] == 1].mean(0)
    batch = torch.stack([torch.from_numpy(t0).float(), torch.from_numpy(t1).float()], 0)
    trainer = DFaSTTrainer(args, task_id=0)
    trainer.load_model()
    trainer.model.eval()
    batch = batch.unsqueeze(1)
    adjacency = trainer.model.fast_p.dca.get_adjacency(batch)
    a0 = adjacency[0, 8, :, :, :].detach().numpy()
    a01 = a0[0]
    a02 = a0[1]
    a03 = a0[2]
    a04 = a0[3]
    a1 = adjacency[1, 8, :, :, :].detach().numpy()
    a11 = a1[0]
    a12 = a1[1]
    a13 = a1[2]
    a14 = a1[3]
    return


if __name__ == '__main__':
    # mva_attention()
    dca_graph()
