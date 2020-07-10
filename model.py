import torch
from torch import nn
from torch.distributions import Beta, Bernoulli, Independent, kl_divergence
from utils import mix_weights


class sb_vae(nn.Module):
    def __init__(self, D, K, prior_alpha, hidden_dim, enc_layer=1, dec_layer=1):
        super(sb_vae, self).__init__()
        self.D = D
        self.K = K
        self.prior_alpha = prior_alpha
        self.encoder = nn.Sequential(*(
            [nn.Linear(D, hidden_dim),
            nn.ReLU()] +
            [nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()] * enc_layer +
            [nn.Linear(hidden_dim, 2 * K), nn.Softplus()]
        ))
        self.decoder = nn.Sequential(*(
            [nn.Linear(K, hidden_dim),
            nn.ReLU()] + 
            [nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()] * dec_layer +
            [nn.Linear(hidden_dim, D)]
        ))

    def encode(self, X):
        encoder_out = self.encoder(X)
        alpha, beta = encoder_out[:, :self.K], encoder_out[:, self.K:]
        return (Independent(Beta(alpha, beta), 1),
                alpha, beta)

    def decode(self, pi):
        logits = self.decoder(pi)
        return Independent(Bernoulli(logits=logits), 1)

    def forward(self, X):
        qz_x, alpha, _ = self.encode(X)
        pz = Independent(
            Beta(torch.ones_like(alpha), torch.ones_like(alpha) * self.prior_alpha), 1)
        pi = mix_weights(qz_x.rsample())[:, :-1]
        px_z = self.decode(pi)
        nll = -px_z.log_prob(X).mean()
        kl = kl_divergence(qz_x, pz).mean()
        return nll, kl