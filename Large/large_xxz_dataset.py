from torch.utils.data import Dataset
import numpy as np

class LargeRandomStateMeasurementResultData(Dataset):
    def __init__(self,num_observables=27,num_states=50,num_qubits=20):
        observables = []
        for i in range(0,num_observables):
            tmp = np.load('2qubit/float_observable2'+str(i)+'.npy')
            observables.append(np.array(tmp))

        index_observables = []
        combination_list = np.load('mso/' + str(num_qubits)+'qubit_sequence_list.npy')
        for j in range(0, len(combination_list)):
            for i in range(0,num_observables):
                tmp = np.concatenate((observables[i],combination_list[j]))
                index_observables.append(tmp)
        self.observables = np.array(index_observables)

        values = []
        ratio_space = np.linspace(-2, 2, 41)
        for i in range(0,41):
            for j in range(0,num_states):
                tmp = np.load('randxxz_prob/rand' + str(num_qubits) + 'qubits_' + str(round(ratio_space[i],1)) + '_' + str(j)+'.npy')
                tmp = tmp.reshape(-1, 4)
                values.append(np.array(tmp,dtype=np.float32))
        self.values = np.array(values)

    def __getitem__(self, idx):
        assert idx < len(self.values)
        return self.observables, self.values[idx]
    def __len__(self):
        return len(self.values)

class TestLargeRandomStateMeasurementResultData(Dataset):
    def __init__(self,num_observables=27,num_states=100,num_qubits=20):
        observables = []
        for i in range(0,num_observables):
            tmp = np.load('2qubit/float_observable2'+str(i)+'.npy')
            observables.append(np.array(tmp))

        index_observables = []
        combination_list = np.load('mso/' + str(num_qubits)+'qubit_sequence_list.npy')
        for j in range(0, len(combination_list)):
            for i in range(0,num_observables):
                tmp = np.concatenate((observables[i],combination_list[j]))
                index_observables.append(tmp)
        self.observables = np.array(index_observables)

        values = []
        ratio_space = np.linspace(-2, 2, 41)
        for i in range(0,41):
            for j in range(0,num_states):
                tmp = np.load('randxxz_prob/test_rand' + str(num_qubits) + 'qubits_' + str(round(ratio_space[i],1)) + '_' + str(j)+'.npy')
                tmp = tmp.reshape(-1, 4)
                values.append(np.array(tmp,dtype=np.float32))
        self.values = np.array(values)

    def __getitem__(self, idx):
        assert idx < len(self.values)
        return self.observables, self.values[idx]
    def __len__(self):
        return len(self.values)