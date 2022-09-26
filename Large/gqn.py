import torch
import torch.nn as nn
from representation import RepresentationNetwork
from generator import GeneratorNetwork

class GenerativeQueryNetwork(nn.Module):
    def __init__(self, x_dim, v_dim, r_dim=8, h_dim=64, z_dim=32, L=12):
        super(GenerativeQueryNetwork, self).__init__()
        self.x_dim = x_dim
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.L = L
        self.r_dim = r_dim

        self.generator = GeneratorNetwork(x_dim, v_dim, r_dim, z_dim, h_dim, L)
        self.representation = RepresentationNetwork(x_dim, v_dim, r_dim)

    def forward(self, context_x, context_v, query_x, query_v):
        b, m, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape

        x = context_x.view((-1, *x_dims))
        v = context_v.view((-1, *v_dims))

        phi = self.representation(x, v)

        _, *phi_dims = phi.shape
        phi = phi.view((b, m, *phi_dims))

        r = torch.mean(phi, dim=1)

        x_mu, kl = self.generator(query_x, query_v, r)

        return (x_mu, r, kl)

    def sample(self, context_x, context_v, query_v):
        batch_size, n_views, _ = context_x.shape

        _, _, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape

        x = context_x.view((-1, *x_dims))
        v = context_v.view((-1, *v_dims))

        phi = self.representation(x, v)

        _, *phi_dims = phi.shape
        phi = phi.view((batch_size, n_views, *phi_dims))

        r = torch.mean(phi, dim=1)

        x_mu = self.generator.sample(query_v, r)
        return x_mu, r, phi
