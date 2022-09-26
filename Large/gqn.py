import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from representation import RepresentationNetwork
from generator import GeneratorNetwork


class GenerativeQueryNetwork(nn.Module):
    """
    Generative Query Network (GQN) similar as described
    in "Neural scene representation and rendering"

    :param x_dim: dimension of expectation = 1
    :param v_dim: dimensions of observable vector
    :param r_dim: dimension of representation
    :param z_dim: dimension of latent variable
    :param h_dim: dimension of hidden dimensions
    :param L: Number of layers in which latent variables would be sequentially refined
    """
    def __init__(self, x_dim, v_dim, r_dim=8, h_dim=64, z_dim=32, L=12):
        super(GenerativeQueryNetwork, self).__init__()
        self.x_dim = x_dim
        self.v_dim = v_dim
        # self.num_pqubits = num_pqubits
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.L = L
        self.r_dim = r_dim

        self.generator = GeneratorNetwork(x_dim, v_dim, r_dim, z_dim, h_dim, L)
        self.representation = RepresentationNetwork(x_dim, v_dim, r_dim)

    def forward(self, context_x, context_v, query_x, query_v):
        """
        Forward through the GQN.

        :param x: batch of context expectations [b, m, 1]
        :param v: batch of context observables [b, m, v_dim]
        :param x_q: batch of query expectations [b, m', 1]
        :param v_q: batch of query observables [b, m', v_dim]
        """
        # Merge batch and view dimensions.
        b, m, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape

        x = context_x.view((-1, *x_dims))
        v = context_v.view((-1, *v_dims))

        # representation generated from input expectations
        # and corresponding observables
        phi = self.representation(x, v)

        # Seperate batch and view dimensions
        _, *phi_dims = phi.shape
        phi = phi.view((b, m, *phi_dims))

        # sum over view representations
        r = torch.mean(phi, dim=1)

        # Use random (expectation, observable) pair in batch as query
        x_mu, kl = self.generator(query_x, query_v, r)

        # Return reconstruction and query observable
        # for computing error
        return (x_mu, r, kl)

    def sample(self, context_x, context_v, query_v):
        """
        Sample from the network given some context and viewpoint.

        :param context_x: set of context expectations to generate representation
        :param context_v: observables of `context_x`
        :param query_v: observables to generate expectation
        :param sigma: pixel variance
        """
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
