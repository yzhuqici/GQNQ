import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import random

from gqn import GenerativeQueryNetwork
from Ising_mix_dataset import StateMeasurementResultData
from Ising_mix_dataset import TestStateMeasurementResultData

# Data
test_flag = 1
num_bits = 6
split_ratio = 0.9
num_states = 50
num_test_states = 10
num_W_states = num_GHZ_states = 900
num_test_W_states = num_test_GHZ_states = 100
num_observables = 729
train_ds = StateMeasurementResultData(num_observables, num_states,num_W_states,num_GHZ_states)
test_ds = TestStateMeasurementResultData(num_observables, num_test_states,num_test_W_states,num_test_GHZ_states)
torch.manual_seed(42)
train_loader = DataLoader(train_ds,batch_size=60, shuffle=True)
test_loader = DataLoader(test_ds)

# Model
device_ids=range(torch.cuda.device_count())
r_dim = 32
h_dim = 96
z_dim = 32
model = GenerativeQueryNetwork(x_dim=2**num_bits, v_dim=4**num_bits*2,r_dim=r_dim, h_dim=h_dim, z_dim=z_dim, L=2)
model = torch.nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(device=device_ids[0])

# try:
#     model.load_state_dict(torch.load('Ising_mix_' + str(num_bits) + 'qubit_' + str(r_dim) + '_' + str(h_dim) + '_' + str(z_dim) + '_2_softmax_r2'))
# except:
#     print("NO Load")


sigma = 0.1
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 1
# train_losses = []

best = 0


for i in tqdm(range(0, epochs)):
    print(i)
    test_loss = 0
    train_loss = 0
    refer_loss = 0
    count1 = 0
    count2 = 0

    if test_flag == 0:
        for v, x in train_loader:
            v = v.cuda(device=device_ids[0])
            x = x.cuda(device=device_ids[0])
            batch_size, m, *_ = v.size()
            n_views = int((num_observables-400) * random.random())+400

            indices = list(range(0, m))
            random.shuffle(indices)
            representation_idx, query_idx = indices[:n_views], indices[n_views:]
            context_x, context_v = x[:, representation_idx], v[:, representation_idx]
            query_x, query_v = x[:, query_idx], v[:, query_idx]
            context_x = context_x.float()
            context_v = context_v.float()
            query_x = query_x.float()
            query_v = query_v.float()

            (x_mu, r, kl) = model(context_x, context_v, query_x, query_v)
            nll = -Normal(x_mu, sigma).log_prob(query_x)
            reconstruction = torch.mean(nll.view(batch_size, -1), dim=0).sum()
            kld = torch.mean(kl.view(batch_size, -1), dim=0).sum()

            x_mu = torch.relu(x_mu)
            train_loss += torch.mean((torch.sum(torch.mul(torch.sqrt(x_mu), torch.sqrt(query_x)), dim=[2])) ).item()
            count1 += 1
            elbo = reconstruction + kld
            elbo.backward()

            optimizer.step()
            optimizer.zero_grad()
        print(train_loss / count1)

    for v, x in test_loader:
        v = v.cuda(device=device_ids[0])
        x = x.cuda(device=device_ids[0])

        batch_size, m, *_ = v.size()
        n_views = 30
        indices = list(range(0, m))
        random.shuffle(indices)
        representation_idx, query_idx = indices[:n_views], indices[n_views:]
        context_x, context_v = x[:, representation_idx], v[:, representation_idx]
        query_x, query_v = x[:, query_idx], v[:, query_idx]
        context_x = context_x.float()
        context_v = context_v.float()
        query_x = query_x.float()
        query_v = query_v.float()

        test_x = torch.tensor(np.ones(query_x.shape)/(2**6)).cuda(device=device_ids[0])

        x_mu ,  phi, r = model.module.sample(context_x, context_v, query_v)
        x_mu = torch.relu(x_mu)

        test_loss += torch.mean((torch.sum(torch.mul(torch.sqrt(x_mu), torch.sqrt(query_x)), dim=[2])) ).item()
        refer_loss += torch.mean((torch.sum(torch.mul(torch.sqrt(test_x), torch.sqrt(query_x)), dim=[2])) ).item()

        count2 += 1

    print(test_loss / count2)
    print(refer_loss/count2)

    # if test_flag == 0:
    #     torch.save(model.state_dict(), 'Ising_mix_' + str(num_bits) + 'qubit_' + str(r_dim) + '_' + str(h_dim) + '_' + str(z_dim) + '_2_softmax_r2')

