from sklearn.decomposition import PCA
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch

datatype = torch.float64


def ppca(data, dim):
    X = data.numpy()
    pca = PCA()
    pca.fit(X)
    u = pca.components_
    v = pca.explained_variance_
    v = np.where(v < 0, 0, v)
    x_mean = np.mean(X, axis=0)
    x_center = np.zeros(X.shape)
    for i in range(X.shape[1]):
        x_center[:, i] = X[:, i] - x_mean[i]
    X_tem = np.matmul(x_center, u)
    a = X_tem[:, 0:dim]
    b = np.diag(1 / np.sqrt(v[0:dim]))
    new_X = np.matmul(a, b)
    return torch.from_numpy(new_X).to(datatype)


if __name__ == '__main__':
    dims = 8
    data = loadmat("./../datasets/toy1.mat")
    Y = data["Y"]
    a = torch.from_numpy(Y).to(datatype)
    b = a.numpy()
    X = ppca(a, dims)
    # print(a.shape)
    # print(b.shape)
    plt.plot(X)
    plt.show()
    print(X.shape)