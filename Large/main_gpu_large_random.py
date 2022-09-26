import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

import numpy as np
from tqdm import tqdm
import random
import itertools



from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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
# torch.manual_seed(42)
# train_ds, test_ds = random_split(ds, [int(split_ratio*len(ds)),len(ds)-int(split_ratio*len(ds))])
# test_indices = test_ds.indices
# np.save("10qubit_ground_test_indices_partial2",test_indices)
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
try:
    model.load_state_dict(torch.load('Ising_' + str(Nsites) + 'qubit_partial2_'+str(r_dim)+'_'+str(h_dim)+'_'+str(z_dim)+'_2_softmax_sequence_random_all'))
    print("Total number of param in Model is ", sum(x.numel() for x in model.parameters()))
except:
    print("NO load")
# torch.save(model.state_dict(), 'GHZ_state_6qubit_9_0.1pi_32_32_16_2_softmax_good')

sigma = 0.1
lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 20
# train_losses = []



for i in tqdm(range(0, epochs)):
    test_losses = []
    print(i)
    test_loss = 0
    test_loss2 = 0
    train_loss = 0
    refer_loss = 0
    count1 = 0
    count2 = 0
    svalue_count = 0

    if test_flag == 0:
        for v, x in train_loader:
            # v = v.reshape([v.shape[0], v.shape[1], v.shape[2] * v.shape[3]])
            # v_real = v.real
            # v_imag = v.imag
            # v = torch.cat([v_real, v_imag], dim=2)

            v = v.cuda(device=device_ids[0])
            x = x.cuda(device=device_ids[0])
            # print(v.shape)
            # print(x.shape)

            # Sample random number of views for a scene
            batch_size, m, *_ = v.size()
            # print(m)
            n_views = 50
            # n_views = 50
            #         indices = torch.arange(0,m,dtype=torch.long)
            #         print(indices)

            # indices = torch.randperm(m)
            # representation_idx, query_idx = indices[:n_views], indices[n_views:]

            indices = list(range(0, m))
            random.shuffle(indices)
            # representation_idx, query_idx = [0]+indices[:n_views-1], indices[n_views-1:]
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
            #         reconstruction = torch.abs((x_mu - query_x).mean())
            kld = torch.mean(kl.view(batch_size, -1), dim=0).sum()

            # print(torch.abs((x_mu - query_x)))
            # print("******")
            # print(x.shape)
            # print(query_x.shape)
            x_mu = torch.relu(x_mu)
            # print(x_mu)
            # x1 = torch.cat((x_mu,context_x),1)
            # x2 = torch.cat((query_x,context_x),1)
            # train_loss += torch.mean((torch.sum(torch.mul(torch.sqrt(x1), torch.sqrt(x2)), dim=[2]))**2).item()
            train_loss += torch.mean((torch.sum(torch.mul(torch.sqrt(x_mu), torch.sqrt(query_x)), dim=[2]))).item()
            # train_loss += torch.abs((x_mu - query_x)).mean()
            count1 += 1
            #         print(r)
            # print(x_mu)
            #         print(query_x)
            #         print(torch.abs((x_mu - query_x)).mean())
            #         print(kld)
            elbo = reconstruction + kld
            #         print(elbo)
            #         print('---------------')
            elbo.backward()

            optimizer.step()
            optimizer.zero_grad()
        # train_losses.append(train_loss / count1)
        print(train_loss / count1)

    for v, x in test_loader:
        #         print(v.shape)
        # v = v.reshape([v.shape[0], v.shape[1], v.shape[2] * v.shape[3]])
        # v_real = v.real
        # v_imag = v.imag
        # v = torch.cat([v_real, v_imag], dim=2)
        #         print(v.shape)
        #             # Sample random number of views for a scene

        v = v.cuda(device=device_ids[0])
        x = x.cuda(device=device_ids[0])

        batch_size, m, *_ = v.size()
        # print(m)
        # n_views = int((num_observables-1) * random.random())+1
        n_views = 30
        # print(n_views)
        # n_views = 50
        # indices = torch.arange(0,m,dtype=torch.long)
        # indices = torch.randperm(m)
        indices = list(range(0, m))
        random.shuffle(indices)
        representation_idx, query_idx = indices[:n_views], indices[n_views:]
        # representation_idx, query_idx = [0]

        # representation_idx, query_idx = indices[:n_views], indices[n_views:]
        context_x, context_v = x[:, representation_idx], v[:, representation_idx]
        query_x, query_v = x[:, query_idx], v[:, query_idx]
        context_x = context_x.float()
        context_v = context_v.float()
        query_x = query_x.float()
        query_v = query_v.float()

        test_x = torch.tensor(np.ones(query_x.shape)/(2**num_bits)).cuda(device=device_ids[0])

        x_mu, r, phi = model.module.sample(context_x, context_v, query_v)
        x_mu = torch.relu(x_mu)
        # np.save("test",x_mu.detach().cpu().float())

        # x1 = torch.cat((x_mu, context_x), 1)
        # x2 = torch.cat((query_x, context_x), 1)
        # x3 = torch.cat((test_x, context_x), 1)
        # test_loss += torch.mean((torch.sum(torch.mul(torch.sqrt(x1), torch.sqrt(x2)), dim=[2])) ** 2).item()
        # refer_loss += torch.mean((torch.sum(torch.mul(torch.sqrt(x3), torch.sqrt(x2)), dim=[2])) ** 2).item()
        # np.save("test_x_mu",x_mu.detach().cpu())
        # np.save("test_query_x", query_x.detach().cpu())
        test_losses.append(torch.mean((torch.sum(torch.sqrt(torch.mul(x_mu, query_x)), dim=[2]))).item())
        tmp = (torch.sum(torch.mul(torch.sqrt(x_mu), torch.sqrt(query_x)), dim=[2]))
        sorted, indices = torch.sort(tmp)
        # print(tmp.shape)

        # test_loss += torch.mean((torch.sum(torch.mul(torch.sqrt(x_mu), torch.sqrt(query_x)), dim=[2])) ** 2).item()
        test_loss += torch.mean(sorted).item()
        test_loss2 += torch.mean(torch.abs(x_mu-query_x))
        refer_loss += torch.mean((torch.sum(torch.mul(torch.sqrt(test_x), torch.sqrt(query_x)), dim=[2])) ).item()

        count2 += 1

    #     print('------------')

    print(test_loss / count2)
    print(test_loss2 / count2)
    print(refer_loss/count2)

    # np.save(str(Nsites)+"qubit_large_random_all",test_losses)
    if test_flag == 0:
        torch.save(model.state_dict(), 'Ising_' + str(Nsites) + 'qubit_partial2_'+str(r_dim)+'_'+str(h_dim)+'_'+str(z_dim)+'_2_softmax_sequence_random_all')







