import torch
from torch.utils.data import DataLoader
from torch.distributions import Normal, kl_divergence

import numpy as np
from tqdm import tqdm
import random

from gqn import GenerativeQueryNetwork
from large_dataset import LargeRandomStateMeasurementResultData, TestLargeRandomStateMeasurementResultData

# Data
Nsites = 20
test_flag = 1
num_bits = 2
split_ratio = 0.9
num_states = 50
num_test_states = 10
num_observables = 9
train_ds = LargeRandomStateMeasurementResultData(num_observables, num_states,Nsites)
test_ds = TestLargeRandomStateMeasurementResultData(num_observables, num_test_states,Nsites)
train_loader = DataLoader(train_ds,batch_size=20)
test_loader = DataLoader(test_ds)

# Model
device_ids=range(torch.cuda.device_count())
r_dim = 24
h_dim = 48
z_dim = 24
model = GenerativeQueryNetwork(x_dim=2**num_bits, v_dim=4**num_bits*2+2,r_dim=r_dim, h_dim=h_dim, z_dim=z_dim, L=2)
model = torch.nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(device=device_ids[0])
# try:
#     model.load_state_dict(torch.load('Ising_' + str(Nsites) + 'qubit_'+str(r_dim)+'_'+str(h_dim)+'_'+str(z_dim)+'_2_softmax'))
#     print("Total number of param in Model is ", sum(x.numel() for x in model.parameters()))
# except:
#     print("NO load")

sigma = 0.1
lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 20

for i in tqdm(range(0, epochs)):
    print(i)
    test_loss = 0
    train_loss = 0
    count1 = 0
    count2 = 0

    if test_flag == 0:
        for v, x in train_loader:
            v = v.cuda(device=device_ids[0])
            x = x.cuda(device=device_ids[0])
            batch_size, m, *_ = v.size()

            n_views = 50

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
            train_loss += torch.mean((torch.sum(torch.mul(torch.sqrt(x_mu), torch.sqrt(query_x)), dim=[2]))).item()
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

        x_mu, r, phi = model.module.sample(context_x, context_v, query_v)
        x_mu = torch.relu(x_mu)

        tmp = (torch.sum(torch.mul(torch.sqrt(x_mu), torch.sqrt(query_x)), dim=[2]))
        sorted, indices = torch.sort(tmp)
        test_loss += torch.mean(sorted).item()

        count2 += 1

    print(test_loss / count2)

    # if test_flag == 0:
    #     torch.save(model.state_dict(), 'Ising_' + str(Nsites) + 'qubit'+str(r_dim)+'_'+str(h_dim)+'_'+str(z_dim)+'_2_softmax')





