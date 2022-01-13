import torch
from kernel_ard import RBFard

v = torch.tensor([0.3])
l = torch.tensor([0.1, 2])
X = torch.tensor([[0.1, 2], [3, 4], [5, 6]])
Y = torch.tensor([[0.1, 2], [7, 8]])
n = 2

k = RBFard(input_dim=n, variance=v, lengthscale=l)
print(k(Y, X))
