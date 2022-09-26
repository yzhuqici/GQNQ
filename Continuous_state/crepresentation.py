import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationNetwork(nn.Module):
    def __init__(self, x_dim, v_dim, r_dim):
        super(RepresentationNetwork, self).__init__()
        self.r_dim = k = r_dim
        self.linear1 = nn.Linear(v_dim, k)
        self.linear2 = nn.Linear(x_dim, k)
        self.linear3 = nn.Linear(2*k, k)
        self.linear4 = nn.Linear(k , k)

    def forward(self, x, v):
        v = F.relu(self.linear1(v))
        x = F.relu(self.linear2(x))
        merge = torch.cat([x, v], dim=1)
        r = F.relu(self.linear3(merge))
        r = F.relu(self.linear4(r))
        return r