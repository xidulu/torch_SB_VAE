# Torch SB VAE
Implementation of stick breaking VAE with pytorch

Different from the origin paper, q(z | x) follows Beta distribution, since the pahtwise gradient for Beta distribution is available in torch.distributions.Beta.

TODO:
- Parameter tuning to reproduce the paper results
