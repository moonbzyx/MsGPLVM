import torch
import numpy as np

# a = [i for i in range(1, 5)]
a = torch.tensor((range(1, 5))).reshape(2, 2) * 1.0
b = torch.tensor((range(5, 9))).reshape(2, 2) * 1.0
c = torch.tensor((range(11, 15))).reshape(-1, 1) * 1.0
aa = torch.diag(a)
bb = torch.diag(b)

# print(aa)
# print(bb)
# print(c)
# d = aa @ bb @ c
# print(d)
# m1 = a.reshape((2, 3))
# print(m1)
# print(m1.shape)
# m2 = m1.sum(dim=-1)
# print(m2)
# print(m2.shape)
test = torch.einsum('ij,ij->ij', [a, b])
print(test)
