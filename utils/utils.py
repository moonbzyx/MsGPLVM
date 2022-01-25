from unittest import result
import numpy as np
from sklearn.cluster import KMeans
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
# from ppca import ppca
from .ppca import ppca
import random

datatype = torch.float64


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


class statistics_psi():
    def __init__(self, phi, sigma, w, mu, S):
        self.phi = phi
        self.sigma = sigma**2
        self.w = w
        self.mu = mu
        self.S = S
        self.N, self.D, self.L = self.phi.shape

    def psi_0(self):
        phi = self.phi.permute([2, 1, 0]).sum(dim=-1)
        variance = self.sigma.unsqueeze(-1).expand([self.L, self.D])
        psi = torch.einsum('ij,ij->ij', [phi, variance])
        return psi

    def psi_1(self):
        pass

    def psi_2(self):
        pass


if __name__ == '__main__':
    data = loadmat("./../datasets/toy1.mat")
    Y = data["Y"]
    X = torch.from_numpy(Y).to(datatype)
    init_X = ppca(X, 8)
    xu = init_inducing(init_X, 50, 3)
    print(init_X.shape)
    print(xu[0])
