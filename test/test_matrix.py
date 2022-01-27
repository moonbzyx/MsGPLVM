import torch

a = torch.arange(2 * 2)
b = torch.arange(2 * 2).reshape(([2, 2])) * 0.1
# a = a.unsqueeze(dim=2)
# a = a.unsqueeze(1).expand(([2, 3, 4]))
# c = a.expand([2, 3, 4])
c = a
c = a.unsqueeze(-1).expand([4, 3])
dd = torch.eye(2, 3)
print(dd)