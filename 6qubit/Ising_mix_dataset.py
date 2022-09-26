import torch
from torch.utils.data import Dataset
import numpy as np

class StateMeasurementResultData(Dataset):
    def __init__(self,num_observables=729,num_states=50, num_W_states = 1000,num_GHZ_states=1000):
        observables = []
        for i in range(0,num_observables):
            tmp = np.load('6qubit/float_observable6'+str(i)+'.npy')
            observables.append(np.array(tmp))
        self.observables = np.array(observables)

        ratio_space = np.linspace(-2,2,41)
        values = []
        for i in range(0, 41):
            for j in range(0, num_states):
                tmp = np.load('6qubit/xxz_ground_state_6qubit_probs'+ str(round(ratio_space[i],1))+'_'+str(j)+'.npy')
                tmp = tmp.reshape(-1, 2**6)
                values.append(np.array(tmp, dtype=np.float32))

        for i in range(0, 41):
            for j in range(0, num_states):
                tmp = np.load('6qubit/Ising_ground_state_6qubit_probs'+ str(round(ratio_space[i],1))+'_'+str(j)+'.npy')
                tmp = tmp.reshape(-1, 2**6)
                values.append(np.array(tmp, dtype=np.float32))

        for j in range(0,int(num_GHZ_states)):
            tmp = np.load('6qubit/probs_6qubit_9_0.2pi'+str(j)+'.npy')
            values.append(np.array(tmp,dtype=np.float32))
        for j in range(0, int(num_W_states)):
            tmp = np.load('6qubit/W_probs_6qubit_9_0.2pi' + str(j) + '.npy')
            values.append(np.array(tmp, dtype=np.float32))
        self.expectation_values = np.array(values)

    def __getitem__(self, idx):
        assert idx < len(self.expectation_values)
        return self.observables, self.expectation_values[idx]

    def __len__(self):
        return len(self.expectation_values)
class TestStateMeasurementResultData(Dataset):
    def __init__(self,num_observables=27,num_states=10, num_W_states = 100,num_GHZ_states=100):
        observables = []
        for i in range(0,num_observables):
            tmp = np.load('6qubit/float_observable6'+str(i)+'.npy')
            observables.append(np.array(tmp))
        self.observables = np.array(observables)

        ratio_space = np.linspace(-2,2,41)
        values = []
        for i in range(0, 41):
            for j in range(0, num_states):
                tmp = np.load('6qubit/xxz_ground_state_6qubit_probs_test'+ str(round(ratio_space[i],1))+'_'+str(j)+'.npy')
                tmp = tmp.reshape(-1, 2**6)
                values.append(np.array(tmp, dtype=np.float32))
        for i in range(0, 41):
            for j in range(0, num_states):
                tmp = np.load('6qubit/Ising_ground_state_6qubit_probs_test'+ str(round(ratio_space[i],1))+'_'+str(j)+'.npy')
                tmp = tmp.reshape(-1, 2**6)
                values.append(np.array(tmp, dtype=np.float32))

        for j in range(999,999-int(num_GHZ_states),-1):
            tmp = np.load('6qubit/probs_6qubit_9_0.2pi'+str(j)+'.npy')
            values.append(np.array(tmp,dtype=np.float32))
        for j in range(999, 999-int(num_W_states),-1):
            tmp = np.load('6qubit/W_probs_6qubit_9_0.2pi' + str(j) + '.npy')
            values.append(np.array(tmp, dtype=np.float32))
        self.expectation_values = np.array(values)

    def __getitem__(self, idx):
        assert idx < len(self.expectation_values)
        return self.observables, self.expectation_values[idx]

    def __len__(self):
        return len(self.expectation_values)


