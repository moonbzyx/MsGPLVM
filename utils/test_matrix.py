import torch

a = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]])
b = a * 0.1
c = torch.stack([a, b], dim=0)
c = c.permute([0, 2, 1])
d = c.permute([2, 1, 0])

# print(c.shape)
# print(c)
# print(d.shape)
# print(d)

# mm = torch.diag(torch.tensor([1, 2, 3])) * 2.0
# print(torch.linalg.inv(mm))

mm = torch.arange((2 * 3)).reshape(([2, 3]))

print(mm.shape)
mmm = mm.unsqueeze(0)
print(mmm.shape)
print(mmm)
mmmm = mmm.expand([3, 2, 3])
print(mmmm.shape)
print(mmmm)
