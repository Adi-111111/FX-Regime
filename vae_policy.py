import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, deterministic=False):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu if deterministic else self.reparameterise(mu, logvar)

        reconstruction = self.decoder(z)
        raw_policy = self.policy_head(z)

        weight = torch.tanh(raw_policy[:, 0:1])
        sl_mult = torch.sigmoid(raw_policy[:, 1:2]) * 2.0 + 1.5
        tp_mult = torch.sigmoid(raw_policy[:, 2:3]) * 2.5 + 1.5

        return reconstruction, mu, logvar, weight, sl_mult, tp_mult

def institutional_sharpe_loss(reconstruction, x, mu, logvar, weights, barrier_labels,
                               price_returns, temporal_features=None,
                               spread=0.00012, comm=0.00003,
                               lambd_time=100.0, t_cutoff=0.91):

    recon_loss = F.mse_loss(reconstruction, x, reduction='mean')

    # sum over latent dims, mean over batch -- keeps this term on the same
    # scale as recon_loss regardless of batch size, unlike a raw batch-wide sum
    kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld_loss = kld_per_sample.mean()

    target_direction = torch.tensor(barrier_labels, dtype=torch.float32).unsqueeze(1)
    alignment_loss = torch.mean(torch.relu(-weights[:len(target_direction)] * target_direction))

    portfolio_returns = weights[:-1] * price_returns[1:]
    transaction_costs = torch.where(torch.abs(weights[:-1]) > 0.05, spread + comm, 0.0)
    net_returns = portfolio_returns - transaction_costs

    expected_return = torch.mean(net_returns)
    volatility = torch.std(net_returns) + 1e-6
    sharpe_loss = -(expected_return / volatility)

    time_penalty = 0.0
    if temporal_features is not None:
        time_penalty = torch.mean(torch.abs(weights[:-1]) * torch.relu(temporal_features[:-1] - t_cutoff))

    return (recon_loss * 0.1) + (0.5 * kld_loss) + (50.0 * sharpe_loss) + \
           (150.0 * alignment_loss) + (lambd_time * time_penalty)
