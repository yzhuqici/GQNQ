from torch.utils.data import Dataset
import numpy as np
import random

class CatGaussData(Dataset):
    def __init__(self,num_observables=30,num_states=2500):
        self.observables = np.load('phis.npy')
        indices1 = list(range(0,18750))
        indices2 = list(range(0,12150))
        random.shuffle(indices1)
        random.shuffle(indices2)
        self.values = np.concatenate((np.load('cat_probs.npy')[indices2[0:10000]],np.load('gaussian_probs.npy')[indices1[0:10000]]))
    def __getitem__(self, idx):
        assert idx < len(self.values)
        return self.observables, self.values[idx]

    def __len__(self):
        return len(self.values)

class GaussData(Dataset):
    def __init__(self,num_observables=30,num_states=2500):
        self.observables = np.load('phis.npy')
        self.values = np.load('gaussian_probs.npy')
    def __getitem__(self, idx):
        assert idx < len(self.values)
        return self.observables, self.values[idx]

    def __len__(self):
        return len(self.values)

class CatStateData(Dataset):
    def __init__(self,num_observables=30,num_states=2500):
        self.observables = np.load('phis.npy')
        self.values = np.load('cat_probs.npy')
    def __getitem__(self, idx):
        assert idx < len(self.values)
        return self.observables, self.values[idx]

    def __len__(self):
        return len(self.values)

class GKPStateData(Dataset):
    def __init__(self,num_observables=30,num_states=2500):
        self.observables = np.load('phis.npy')
        self.values = np.load('gkp_probs.npy')
    def __getitem__(self, idx):
        assert idx < len(self.values)
        return self.observables, self.values[idx]

    def __len__(self):
        return len(self.values)

class MixStateData(Dataset):
    def __init__(self,num_observables=30,num_states=2500):
        self.observables = np.load('phis.npy')
        indices1 = list(range(0, 18750))
        indices2 = list(range(0, 12150))
        indices3 = list(range(0, 11025))
        random.shuffle(indices1)
        random.shuffle(indices2)
        random.shuffle(indices3)
        self.values = np.concatenate((np.load('cat_probs.npy')[indices2[0:10000]],np.load('gaussian_probs.npy')[indices1[0:10000]],np.load('gkp_probs.npy')[indices3[0:10000]]))
    def __getitem__(self, idx):
        assert idx < len(self.values)
        return self.observables, self.values[idx]

    def __len__(self):
        return len(self.values)
