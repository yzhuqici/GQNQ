import torch
import torch.nn as nn
import torch.nn.functional as F

class RepresentationNetwork(nn.Module):
    def __init__(self, x_dim, v_dim, r_dim):

        super(RepresentationNetwork, self).__init__()
        self.r_dim = k = r_dim
        self.v_dim = v_dim
        self.linear1 = nn.Linear(v_dim-3, k)
        self.linear2 = nn.Linear(x_dim, k)
        self.linear3 = nn.Linear(3, k)
        self.linear4 = nn.Linear(3*k, k)
        self.linear5 = nn.Linear(k , k)
        # self.linear6 = nn.Linear(2*k, k)

    def forward(self, x, v):
        v1 = F.relu(self.linear1(v[:,0:self.v_dim-3]))
        x = F.relu(self.linear2(x))
        v2 = F.relu(self.linear3(v[:,self.v_dim-3:]))
        merge = torch.cat([x,v1,v2], dim=1)
        r = F.relu(self.linear4(merge))
        r = F.relu(self.linear5(r))
        # r = F.relu(self.linear6(r))
        return r