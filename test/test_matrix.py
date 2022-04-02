import torch
from pyro.distributions.util import eye_like

a = torch.arange(2 * 2)
b = torch.tensor([[1, 2, 3], [4, 5, 6]], device=torch.device('cuda'))
# a = a.unsqueeze(dim=2)
# a = a.unsqueeze(1).expand(([2, 3, 4]))
# c = a.expand([2, 3, 4])
c = b[0]
print(id(c))
print(id(c) - id(b))
print(id(a) - id(b))
print(id(b[0]))
print(b)