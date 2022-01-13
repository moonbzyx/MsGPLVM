import pyro.distributions as dist
import pyro
import torch

# ----------to substitute the for-loop-------------------
# mean = mean.unsqueeze(-1).expand((len(ruggedness),))
# sigma = sigma.unsqueeze(-1).expand((len(ruggedness),))
# return pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)
# ----------to substitute the for-loop-------------------

data = torch.tensor([[1, 2], [3, 4]])
m1 = torch.zeros(2)
m2 = torch.ones(2) * 10
k1 = torch.eye((2)) * 0.01
k2 = torch.eye((2)) * 100
a = pyro.sample('a', dist.MultivariateNormal(m1, k1))
# with pyro.plate('N', 2):
#     aa = pyro.sample('aa', dist.MultivariateNormal(m1, k1))

b = pyro.sample('b', dist.MultivariateNormal(m2, k2))
m3 = torch.stack([m1, m2], dim=0)
k3 = torch.stack([k1, k2], dim=0)
# c = pyro.sample('c', dist.MultivariateNormal(torch.zeros(2, 3, 2, 2), k3))
d = pyro.sample('d', dist.MultivariateNormal(m3, k3))
# print('\nm1', m1)
# print('\nk1', k1)
# print('\nm2', m2)
# print('\nk2', k2)
# print('\nm3.shape', m3.shape)
# print('\nm3', m3)
# print('\nk3', k3)
# print('\na.shape', a.shape)
# print('\na', a)
# print('\na.shape', aa.shape)
# print('\naa', aa)
# print(a, "\n", b, "\n", c)
# dd = torch.tensor([[1, 2], [3, 4]])
# d = [i for i in dd.shape]
# d.append(2)
print(d)
# print(ddsh[0])
# mean = dd.unsqueeze((-1)).expand(d.append(2))
# print(mean.shape)
# print(mean[:, :, 1])
