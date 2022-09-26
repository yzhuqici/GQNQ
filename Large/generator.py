import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class LSTMCell(nn.Module):
    """
    long short-term memory (LSTM) cell
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(LSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.forget = nn.Linear(in_channels, out_channels)
        self.input  = nn.Linear(in_channels, out_channels)
        self.output = nn.Linear(in_channels, out_channels)
        self.state  = nn.Linear(in_channels, out_channels)


    def forward(self, input, states):
        """
        Send input through the cell.

        :param input: input to send through
        :param states: (hidden, cell) pair of internal state
        :return new (hidden, cell) pair
        """
        (hidden, cell) = states

        forget_gate = torch.sigmoid(self.forget(input))
        input_gate  = torch.sigmoid(self.input(input))
        output_gate = torch.sigmoid(self.output(input))
        state_gate  = torch.sigmoid(self.state(input))

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return hidden, cell


"""
The generator network is LSTM like network latent variables to learn
the distributions.
"""


class GeneratorNetwork(nn.Module):
    """
    x_dim -> dimension of expectation = 1
    v_dim -> dimensions of observable vector
    r_dim -> dimension of representation
    z_dim -> dimension of latent variable
    h_dim -> dimension of hidden dimensions
    L -> Number of layers in which latent variables would be sequentially refined
    """

    def __init__(self, x_dim=1, v_dim=128, r_dim=32, z_dim=32, h_dim=64, L=3):
        super(GeneratorNetwork, self).__init__()
        self.L = L
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Core layers consists of inference and generator layers:
        # Inference layers gives posterior distribution from which we sample latent variable
        # Generator layer gives prior distribution and generates the prediction given input and latent variable
        # Gist is inference and generator behave like variational autoencoders

        self.inference_core = LSTMCell(h_dim + x_dim + v_dim +  r_dim, h_dim)
        self.generator_core = LSTMCell(v_dim + r_dim + z_dim, h_dim)

        # To obtain posterior and prior we use another Linear layers for each
        # Output is 2 x no. of dimensions of latent variable to accomodate mean and std. deviation of
        # the distributions. We just split the output tensor in half to get mean and std. dev which can
        # be used later for sampling
        self.posterior_density = nn.Linear(h_dim, 2 * z_dim)
        self.prior_density = nn.Linear(h_dim, 2 * z_dim)

        # Generative density
        self.observation_density = nn.Linear(h_dim, x_dim)


    def forward(self, x, v, r):
        """
        Attempt to reconstruct expectation x with corresponding
        observable vector v and context representation r.

        :param x: expectation values
        :param v: obsevervables
        :param r: representation for the state
        :return reconstruction of x and kl-divergence
        """
        batch_size, m, _ = x.shape
        kl = 0

        r = r.unsqueeze(1)
        r = r.repeat(1, m, 1)

        # Reset hidden and cell state
        hidden_i = x.new_zeros((batch_size, m, self.h_dim))
        cell_i = x.new_zeros((batch_size, m, self.h_dim))
        hidden_g = x.new_zeros((batch_size, m, self.h_dim))
        cell_g = x.new_zeros((batch_size, m, self.h_dim))

        # Canvas for updating
        u = x.new_zeros((batch_size, m, self.h_dim))

        for l in range(self.L):
            # Prior factor (eta π network)
            p_mu, p_std = torch.chunk(self.prior_density(hidden_g), 2, dim=2)
            prior_distribution = Normal(p_mu, F.softplus(p_std))

            # Inference state update
            inference = self.inference_core
            hidden_i, cell_i = inference(torch.cat([hidden_g, x, v, r], dim=2), [hidden_i, cell_i])

            # Posterior factor (eta e network)
            q_mu, q_std = torch.chunk(self.posterior_density(hidden_i), 2, dim=2)
            posterior_distribution = Normal(q_mu, F.softplus(q_std))

            # Posterior sample
            z = posterior_distribution.rsample()

            # Generator state update
            generator = self.generator_core
            hidden_g, cell_g = generator(torch.cat([z, v, r], dim=2), [hidden_g, cell_g])

            # Calculate u
            u = hidden_g + u

            # Calculate KL-divergence
            kl += kl_divergence(posterior_distribution, prior_distribution)

        x_mu = self.observation_density(u)
        # return torch.tanh(x_mu),kl
        return torch.softmax(x_mu,dim=2), kl


    def sample(self, v, r):
        """
        Sample from the prior distribution to generate new expectation,
        given any arbritary observable and state representation.
        v -> observables
        r -> representation
        """
        batch_size, m, _ = v.shape

        # Reshape the representation
        r = r.unsqueeze(1)
        r = r.repeat(1, m, 1)

        hidden_g = v.new_zeros((batch_size, m, self.h_dim))
        cell_g = v.new_zeros((batch_size, m, self.h_dim))
        u = v.new_zeros((batch_size, m, self.h_dim))

        for _ in range(self.L):
            p_mu, p_log_std = torch.chunk(self.prior_density(hidden_g), 2, dim=2)
            prior_distribution = Normal(p_mu, F.softplus(p_log_std))

            # Prior sample
            z = prior_distribution.sample()

            # Calculate u
            hidden_g, cell_g = self.generator_core(torch.cat([z, v, r], dim=2), [hidden_g, cell_g])
            u = hidden_g + u

        x_mu = self.observation_density(u)
        return torch.softmax(x_mu,dim=2)
