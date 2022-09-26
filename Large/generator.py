import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

class LSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.forget = nn.Linear(in_channels, out_channels)
        self.input  = nn.Linear(in_channels, out_channels)
        self.output = nn.Linear(in_channels, out_channels)
        self.state  = nn.Linear(in_channels, out_channels)


    def forward(self, input, states):
        (hidden, cell) = states

        forget_gate = torch.sigmoid(self.forget(input))
        input_gate  = torch.sigmoid(self.input(input))
        output_gate = torch.sigmoid(self.output(input))
        state_gate  = torch.sigmoid(self.state(input))

        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return hidden, cell

class GeneratorNetwork(nn.Module):
    def __init__(self, x_dim=1, v_dim=128, r_dim=32, z_dim=32, h_dim=64, L=3):
        super(GeneratorNetwork, self).__init__()
        self.L = L
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.inference_core = LSTMCell(h_dim + x_dim + v_dim +  r_dim, h_dim)
        self.generator_core = LSTMCell(v_dim + r_dim + z_dim, h_dim)

        self.posterior_density = nn.Linear(h_dim, 2 * z_dim)
        self.prior_density = nn.Linear(h_dim, 2 * z_dim)

        self.observation_density = nn.Linear(h_dim, x_dim)


    def forward(self, x, v, r):
        batch_size, m, _ = x.shape
        kl = 0

        r = r.unsqueeze(1)
        r = r.repeat(1, m, 1)

        hidden_i = x.new_zeros((batch_size, m, self.h_dim))
        cell_i = x.new_zeros((batch_size, m, self.h_dim))
        hidden_g = x.new_zeros((batch_size, m, self.h_dim))
        cell_g = x.new_zeros((batch_size, m, self.h_dim))

        u = x.new_zeros((batch_size, m, self.h_dim))

        for l in range(self.L):
            p_mu, p_std = torch.chunk(self.prior_density(hidden_g), 2, dim=2)
            prior_distribution = Normal(p_mu, F.softplus(p_std))

            inference = self.inference_core
            hidden_i, cell_i = inference(torch.cat([hidden_g, x, v, r], dim=2), [hidden_i, cell_i])

            q_mu, q_std = torch.chunk(self.posterior_density(hidden_i), 2, dim=2)
            posterior_distribution = Normal(q_mu, F.softplus(q_std))

            z = posterior_distribution.rsample()

            generator = self.generator_core
            hidden_g, cell_g = generator(torch.cat([z, v, r], dim=2), [hidden_g, cell_g])

            u = hidden_g + u

            kl += kl_divergence(posterior_distribution, prior_distribution)

        x_mu = self.observation_density(u)
        return torch.softmax(x_mu,dim=2), kl


    def sample(self, v, r):
        batch_size, m, _ = v.shape

        r = r.unsqueeze(1)
        r = r.repeat(1, m, 1)

        hidden_g = v.new_zeros((batch_size, m, self.h_dim))
        cell_g = v.new_zeros((batch_size, m, self.h_dim))
        u = v.new_zeros((batch_size, m, self.h_dim))

        for _ in range(self.L):
            p_mu, p_log_std = torch.chunk(self.prior_density(hidden_g), 2, dim=2)
            prior_distribution = Normal(p_mu, F.softplus(p_log_std))

            z = prior_distribution.sample()

            hidden_g, cell_g = self.generator_core(torch.cat([z, v, r], dim=2), [hidden_g, cell_g])
            u = hidden_g + u

        x_mu = self.observation_density(u)
        return torch.softmax(x_mu,dim=2)
