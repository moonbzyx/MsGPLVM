import pyro
import torch
import pyro.distributions as dist
from pyro.contrib.gp.parameterized import Parameterized
import numpy as np


class GP_TEST(Parameterized):
    X = torch.tensor([2.0])
    X = pyro.param('X', X)
    print("The original X is :", X)

    def model(self):
        X = pyro.sample('X', dist.Normal(0, 1))

    def guide(self):
        X = pyro.sample('X', dist.Normal(100, 1))


# ---------------list ---> torch.tensor
# a = torch.rand((2, 2))
# b = torch.rand((3, 3))
# l = [a, b]
# ll = tuple(l)
# lll = torch.stack(l, dim=0)
# # print(l)
# print(ll)
# print(lll)

#  ----- var in for ------------
# for i in range(10):
#     tt = i
# print(tt)
# c = torch.tensor(([]))
# c = torch.stack(a)
# print(l)

# --------list of list ----------
ll = [[] for _ in range(3)]
for i in range(3):
    ll[i] = torch.rand((2, 2))
print(ll)