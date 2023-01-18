import argparse

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from src.gen_traces import TraceGenerator
from src.utils import str2bool
from src.config2 import Config
import matplotlib.pyplot as plt
import numpy as np

class GenDataLoader(Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        Generator = TraceGenerator(config)
        X_profiling_traintot, Y_profiling_traintot, self.X_attack_test, self.Y_attack_test, self.real_key = Generator.load_traces(config.masking_order)
        self.tracesTrain, self.tracesVal, self.labelsTrain, self.labelsVal = train_test_split(X_profiling_traintot, Y_profiling_traintot, test_size=0.1,
                                                                          random_state=0)
        self.transform = transform

    def choosedataset(self, phase):
        if phase=="train":
            self.X_profiling, self.Y_profiling = np.expand_dims(self.tracesTrain, 1), self.labelsTrain
        elif phase == "val":
            self.X_profiling, self.Y_profiling = np.expand_dims(self.tracesVal, 1), self.labelsVal
        elif phase=="test":
            self.X_profiling, self.Y_profiling = self.X_attack_test, self.Y_attack_test
        else:
            raise("Problem with phase in Dataloader")


    def __len__(self):
        return len(self.X_profiling)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X_profiling[idx]
        sensitive = self.Y_profiling[idx]
        sample = {'trace': trace, 'sensitive': sensitive}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Long_GenDataLoader(Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        Generator = TraceGenerator(config)
        X_profiling_traintot, Y_profiling_traintot, self.X_attack_test, self.Y_attack_test, self.real_key = Generator.load_long_traces()
        self.tracesTrain, self.tracesVal, self.labelsTrain, self.labelsVal = train_test_split(
            X_profiling_traintot, Y_profiling_traintot, test_size=0.1,
            random_state=0)
        self.transform = transform

    def choosedataset(self, phase):
        if phase == "train":
            self.X_profiling, self.Y_profiling = np.expand_dims(self.tracesTrain, 1), self.labelsTrain
        elif phase == "val":
            self.X_profiling, self.Y_profiling = np.expand_dims(self.tracesVal, 1), self.labelsVal
        elif phase == "test":
            self.X_profiling, self.Y_profiling= self.X_attack_test, self.Y_attack_test
        else:
            raise ("Problem with phase in Dataloader")

    def __len__(self):
        return len(self.X_profiling)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X_profiling[idx]
        sensitive = self.Y_profiling[idx]
        sample = {'trace': trace, 'sensitive': sensitive}

        if self.transform:
            sample = self.transform(sample)

        return sample




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        trace, label = sample['trace'], sample['sensitive']

        return {'trace': torch.from_numpy(trace).float(),
                'sensitive': torch.from_numpy(np.array([label]))}


class ToTensor_long(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        trace, label, m_1, m_2 = sample['trace'], sample['sensitive'], sample['m_1'], sample['m_2']

        return {'trace': torch.from_numpy(trace).float(),
                'sensitive': torch.from_numpy(np.array([label])), 'm_1': torch.from_numpy(np.array([m_1])), 'm_2': torch.from_numpy(np.array([m_2]))}


if __name__ == '__main__':
    config = Config(path="../config/")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=config.general.seed, type=int, choices=[i for i in range(100)])
    parser.add_argument("--device", default=config.general.device, type=int, choices=[0, 1, 2, 3])
    parser.add_argument("--logs_tensorboard", default=config.general.logs_tensorboard)
    parser.add_argument("--ascad_database_file", default=config.general.ascad_database_file)

    parser.add_argument("--n_features", default=config.gen_mask_traces.n_features, type=int)
    parser.add_argument("--X_profiling_traces", default=config.gen_mask_traces.X_profiling_traces, type=int)
    parser.add_argument("--X_attack_traces", default=config.gen_mask_traces.X_attack_traces,type=int)
    parser.add_argument("--var_noise", default=config.gen_mask_traces.var_noise)
    parser.add_argument("--k", default=config.gen_mask_traces.k, type = int, choices=[i for i in range(256)])
    parser.add_argument("--masking_order", default=config.gen_mask_traces.masking_order, type = int, choices=[i for i in range(4)])
    parser.add_argument("--nomaskleak", default=config.gen_mask_traces.nomaskleak, type=str2bool, choices=[True, False])

    config = parser.parse_args()
    compose = transforms.Compose([ToTensor()])
    dataloader = GenDataLoader(config, transform=compose)
    dataloader.choosedataset("train")
    # print(dataloader[0]['trace'])
    # print(dataloader[0]['sensitive'])
    """s1 = dataloader[0]['trace']
    s2 = dataloader[1]['trace']
    plt.plot(s1)
    plt.plot(s2)
    plt.show()"""