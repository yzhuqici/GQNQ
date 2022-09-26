import torch
from tqdm import tqdm
import random
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from cgqn import GenerativeQueryNetwork
from cdataset import CatGaussData, GaussData, MixStateData, GKPStateData, CatStateData

ds = CatStateData()
# ds = GKPStateData()
# ds = MixStateData()
# ds = CatGaussData()
# ds = GaussData()

torch.manual_seed(42)
train_ds, test_ds = random_split(ds, [int(0.9*len(ds)),len(ds) - int(0.9*len(ds))])
train_loader = DataLoader(train_ds,batch_size = 20)
test_loader = DataLoader(test_ds)

# Model
device_ids=range(torch.cuda.device_count())
model = GenerativeQueryNetwork(x_dim=100, v_dim=1,r_dim=16, h_dim=32, z_dim=32, L=2)
model = torch.nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(device=device_ids[0])

# print("Total number of param in Model is ", sum(x.numel() for x in model.parameters()))

test_flag = 1
sigma = 0.1
lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 10
max = 1
test_losses = []

for i in tqdm(range(0, epochs)):
    print(i)
    test_loss = 0
    train_loss = 0
    count1 = 0
    count2 = 0
    cfidelity = 0
    if test_flag == 0:
        for v, x in train_loader:
            v = v.cuda(device=device_ids[0])
            x = x.cuda(device=device_ids[0])
            v = v.unsqueeze(dim=2)

            batch_size, m, *_ = v.size()
            n_views = 50 + 250*int(random.random())

            indices = torch.randperm(m)
            representation_idx, query_idx = indices[:n_views], indices[n_views:]

            context_x, context_v = x[:, representation_idx], v[:, representation_idx]
            query_x, query_v = x[:, query_idx], v[:, query_idx]
            context_x = context_x.float()
            context_v = context_v.float()
            query_x = query_x.float()
            query_v = query_v.float()

            (x_mu, r, kl) = model(context_x, context_v, query_x, query_v)
            nll = -Normal(x_mu, sigma).log_prob(query_x)
            reconstruction = torch.mean(nll.view(batch_size, -1), dim=0).mean()
            kld = torch.mean(kl.view(batch_size, -1), dim=0).mean()

            x_mu = torch.relu(x_mu)
            train_loss += torch.abs((x_mu - query_x)).mean()
            count1 += 1

            elbo = reconstruction + kld
            elbo.backward()

            optimizer.step()
            optimizer.zero_grad()
        print(train_loss / count1)


    for v, x in test_loader:
        v = v.cuda(device=device_ids[0])
        x = x.cuda(device=device_ids[0])
        v = v.unsqueeze(dim=2)

        batch_size, m, *_ = v.size()
        n_views = 10

        indices = torch.randperm(m)
        representation_idx, query_idx = indices[:n_views], indices[n_views:]

        context_x, context_v = x[:, representation_idx], v[:, representation_idx]
        query_x, query_v = x[:, query_idx], v[:, query_idx]
        context_x = context_x.float()
        context_v = context_v.float()
        query_x = query_x.float()
        query_v = query_v.float()

        x_mu, r, phi = model.module.sample(context_x, context_v, query_v)
        x_mu = torch.relu(x_mu)
        query_x = torch.relu(query_x)

        x_mu_sum = torch.sum(x_mu,dim=[2])
        x_mu_sum = x_mu_sum.unsqueeze(dim=2)
        x_mu_sum = x_mu_sum.repeat((1, 1, 100))
        x_mu = torch.div(x_mu,x_mu_sum)

        query_x_sum = torch.sum(query_x, dim=[2])
        query_x_sum = query_x_sum.unsqueeze(dim=2)
        query_x_sum = query_x_sum.repeat((1, 1, 100))
        query_x = torch.div(query_x, query_x_sum)

        cfidelity += torch.mean((torch.sum(torch.mul(torch.sqrt(query_x), torch.sqrt(x_mu)), dim=[2]))).item()
        test_loss += torch.abs((x_mu - query_x)).mean()

        count2 += 1

    print(test_loss / count2)
    print(cfidelity/count2)
    if test_flag == 0:
        torch.save(model.state_dict(), "trained_learning_model")
    test_losses.append(cfidelity/count2)











