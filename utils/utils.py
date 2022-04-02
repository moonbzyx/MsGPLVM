from unittest import result
import numpy as np
from sklearn.cluster import KMeans
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
# from ppca import ppca
# from .ppca import ppca
import random
import pyro

datatype = torch.float64


def manual_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    pyro.set_rng_seed(seed)


def init_inducing(init_x, num_inducing, num_gp):
    clu = KMeans(n_clusters=num_gp, init=init_x[:num_gp], n_init=1)
    clu.fit(init_x)
    labels_clu = clu.labels_
    n = sum(labels_clu == 0)
    x_u = torch.zeros(([num_gp, num_inducing, init_x.size(1)]))
    for k in range(num_gp):
        idx = []
        if n > num_inducing:
            idx = [i for i, j in enumerate(labels_clu) if j == 0]
        else:
            idx = [i for i in range(init_x.shape[0])]
            random.shuffle(idx)
        idx = idx[:num_inducing]
        x_u[k] = init_x[idx, :]
    return x_u


def init_w(L, Q, device):
    w = torch.rand([L, Q], device=device)
    w = torch.where(w > 0.5, 0.9999, 0.0001)
    return w


if __name__ == '__main__':
    data = loadmat("./../datasets/toy1.mat")
    # Y = data["Y"]
    # X = torch.from_numpy(Y).to(datatype)
    # init_X = ppca(X, 8)
    # xu = init_inducing(init_X, 50, 3)
    # print(init_X.shape)
    # print(xu[0])
    # print(init_w(3, 8))
