import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import os

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.0')

alpha = torch.rand((3))
v = pyro.sample('v', dist.Dirichlet(alpha))
z = pyro.sample('z', dist.Multinomial(1, v))
print(v)
print(z)
z = torch.nonzero(z).squeeze()

print(z)
