"""
MODEL: THE MSGPLVM MODEL
COPYRIGHT: CHAOJIEMEN 2022-1-4
"""
import torch
import numpy as np
import pyro
from pyro import infer
from pyro.nn import PyroModule, PyroParam, PyroSample
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.contrib.gp.parameterized import Parameterized
from pyro.nn.module import pyro_method
from pyro.optim import Adam
import pyro.ops.stats as stats
from pyro.distributions.util import eye_like
from torch.distributions import transform_to

# from ppca import ppca
from utils.ppca import ppca
from utils.utils import init_inducing
from utils.kernel_ard import RBFard
import matplotlib.pyplot as plt


class MSGPLVM(Parameterized):
    def __init__(self, config, Y, X=None):
        super().__init__()
        # assert isinstance(Y, list)
        assert len(Y) > 0
        assert isinstance(Y[0], torch.Tensor)
        torch.set_default_dtype(torch.float64)

        for key, value in config.items():
            self.__setattr__(key, value)

        self.Y = Y  #  N * D
        self.X = X  #  N * Q

        self.setup_model()

    def setup_model(self):
        if self.X == None:
            self.X = ppca(self.Y, self.Q)
        self.X = self.X * 1e-7
        # print(self.X[0, :4])
        # plt.plot(self.X)
        # plt.show()

        # self.Xu = init_inducing(self.init_X, self.M, self.L)  # M * Q
        xu = []  # L*M*Q
        for i in range(self.L):
            xu.append(stats.resample(self.X.clone(), self.M))
        self.Xu = torch.stack(xu, dim=0)

        self.device = torch.device(self.device)
        self.to(self.device)

    @pyro_method
    def model(self):
        self.set_mode('model')

        # prepare the plates of model
        p_N = pyro.plate('N', self.N)  # num of instance
        # p_M = pyro.plate('M', self.M)  # num of inducing points
        p_D = pyro.plate('D', self.D)  # dim of output(observation)
        # p_Q = pyro.plate('Q', self.Q)  # dim of input(latent)
        # p_L = pyro.plate('L', self.L)  # num of GP components

        # initialize the parameters
        alpha = pyro.param("alpha", lambda: torch.rand(self.L))
        beta = pyro.param("beta", lambda: torch.rand(self.D))
        sigma = pyro.param("sigma", lambda: torch.rand(self.L))
        w = pyro.param("w", lambda: torch.rand(self.L, self.Q))

        # X is different from self.X (ppca from Y)
        X = pyro.sample(
            'X',
            dist.MultivariateNormal(torch.zeros(self.N, self.Q),
                                    torch.eye(self.Q)))

        # Xu is the inducing points, not parameters in 'model function'
        Xu = self.Xu  # L*M*Q

        # epsilon = pyro.sample(
        #     'eps',
        #     dist.MultivariateNormal(torch.zeros(self.N, self.D),
        #                             torch.diag(1.0 / beta)))

        with p_D:
            v = pyro.sample('v', dist.Dirichlet(alpha))
            with p_N:
                z = pyro.sample('z', dist.Multinomial(1, v))

        # calculate parts of the kerneal groups
        k_L_MM = []  # kernels : Xu_Xu-->u
        k_L_NN = []  # kernels : X_X
        k_L_NM = []  # kernels : X_Xu
        k_L_MN = []  # kernels : Xu_X
        for l in range(self.L):
            k = RBFard(input_dim=self.Q, variance=sigma[l], lengthscale=w[l])
            k_L_MM.append(k(Xu[l]))
            k_L_NN.append(k(X))
            k_L_NM.append(k(X, Xu[l]))
            k_L_MN.append(k(Xu[l], X))
        k_L_MM = torch.stack(k_L_MM, dim=0)
        k_L_NN = torch.stack(k_L_NN, dim=0)
        k_L_NM = torch.stack(k_L_NM, dim=0)
        k_L_MN = torch.stack(k_L_MN, dim=0)

        # the inducing outputs, L*D
        # the number of it's feature is M ----> final shape: D*L*M
        u = pyro.sample(
            'u',
            dist.MultivariateNormal(torch.zeros(self.D, self.L, self.M),
                                    k_L_MM))

        # calculate the mean and kernel of 'f ~ GPs'
        mm_L_D = []
        kk_L = []
        for l in range(self.L):
            mm = k_L_NM[l] @ torch.linalg.inv(k_L_MM[l])
            kk = k_L_NN[l] - mm @ k_L_MN[l]
            mm_D = []
            for d in range(self.D):
                mm_d = mm @ u[d, l, :]
                mm_D.append(mm_d)
            mm_D = torch.stack(mm_D, dim=0)
            mm_L_D.append(mm_D)
            kk_L.append(kk)
        mm_L_D = torch.stack(mm_L_D, dim=0)
        kk_L = torch.stack(kk_L, dim=0)

        kk_L_D_ii = []
        for l in range(self.L):
            kk_D_ii = torch.diag(kk_L[l]).unsqueeze(0).expand([self.D, self.N])
            kk_L_D_ii.append(kk_D_ii)
        kk_L_D_ii = torch.stack(kk_L_D_ii, dim=0)
        # print(kk_L_D_ii.shape)

        # select GPs (for L kernels) for every pixel of data
        zz = z.permute([2, 1, 0])
        mm_D_N = torch.einsum('ijk,ijk->ijk', [mm_L_D, zz]).sum(dim=0)
        kk_D_ii = torch.einsum('ijk,ijk->ijk', [kk_L_D_ii, zz]).sum(dim=0)
        kk_D_NN = []
        for d in range(self.D):
            kk_D_NN.append(torch.diag(kk_D_ii[d]))
        kk_D_NN = torch.stack(kk_D_NN, dim=0)

        F = pyro.sample('F', dist.MultivariateNormal(mm_D_N, kk_D_NN))
        Y = pyro.sample('Y',
                        dist.MultivariateNormal(F.t(), torch.diag(1 / beta)),
                        obs=self.Y)

        print('************************************************')
        print("v.shape", v.shape)
        print("z.shape", z.shape)
        print("u.shape", u.shape)
        print("X.shape", X.shape)
        print('F.shape', F.shape)
        print('Y.shape', Y.shape)
        print('************************************************')

        return {'X': X, 'u': u, 'F': F, 'v': v, 'z': z}

    @pyro_method
    def guide(self):
        self.set_mode('guide')

        # prepare the plates of model
        p_N = pyro.plate('N', self.N)  # num of instance
        # p_M = pyro.plate('M', self.M)  # num of inducing points
        p_D = pyro.plate('D', self.D)  # dim of output(observation)
        # p_Q = pyro.plate('Q', self.Q)  # dim of input(latent)
        # p_L = pyro.plate('L', self.L)  # num of GP components

        # initialize the parameters
        alpha = pyro.param("alpha", lambda: torch.rand(self.L))
        beta = pyro.param("beta", lambda: torch.rand(self.D))
        sigma = pyro.param("sigma", lambda: torch.rand(self.L))
        w = pyro.param("w", lambda: torch.rand(self.L, self.Q))

        # X is different from self.X (ppca from Y)
        X = pyro.sample(
            'X',
            dist.MultivariateNormal(torch.zeros(self.N, self.Q),
                                    torch.eye(self.Q)))

        # Xu is the inducing points, not parameters in 'model function'
        Xu = self.Xu  # L*M*Q

        # epsilon = pyro.sample(
        #     'eps',
        #     dist.MultivariateNormal(torch.zeros(self.N, self.D),
        #                             torch.diag(1.0 / beta)))

        with p_D:
            v = pyro.sample('v', dist.Dirichlet(alpha))
            with p_N:
                z = pyro.sample('z', dist.Multinomial(1, v))

        # calculate parts of the kerneal groups
        k_L_MM = []  # kernels : Xu_Xu-->u
        k_L_NN = []  # kernels : X_X
        k_L_NM = []  # kernels : X_Xu
        k_L_MN = []  # kernels : Xu_X
        for l in range(self.L):
            k = RBFard(input_dim=self.Q, variance=sigma[l], lengthscale=w[l])
            k_L_MM.append(k(Xu[l]))
            k_L_NN.append(k(X))
            k_L_NM.append(k(X, Xu[l]))
            k_L_MN.append(k(Xu[l], X))
        k_L_MM = torch.stack(k_L_MM, dim=0)
        k_L_NN = torch.stack(k_L_NN, dim=0)
        k_L_NM = torch.stack(k_L_NM, dim=0)
        k_L_MN = torch.stack(k_L_MN, dim=0)

        # the inducing outputs, L*D
        # the number of it's feature is M ----> final shape: D*L*M
        u = pyro.sample(
            'u',
            dist.MultivariateNormal(torch.zeros(self.D, self.L, self.M),
                                    k_L_MM))

        # calculate the mean and kernel of 'f ~ GPs'
        mm_L_D = []
        kk_L = []
        for l in range(self.L):
            mm = k_L_NM[l] @ torch.linalg.inv(k_L_MM[l])
            kk = k_L_NN[l] - mm @ k_L_MN[l]
            mm_D = []
            for d in range(self.D):
                mm_d = mm @ u[d, l, :]
                mm_D.append(mm_d)
            mm_D = torch.stack(mm_D, dim=0)
            mm_L_D.append(mm_D)
            kk_L.append(kk)
        mm_L_D = torch.stack(mm_L_D, dim=0)
        kk_L = torch.stack(kk_L, dim=0)

        kk_L_D_ii = []
        for l in range(self.L):
            kk_D_ii = torch.diag(kk_L[l]).unsqueeze(0).expand([self.D, self.N])
            kk_L_D_ii.append(kk_D_ii)
        kk_L_D_ii = torch.stack(kk_L_D_ii, dim=0)
        # print(kk_L_D_ii.shape)

        # select GPs (for L kernels) for every pixel of data
        zz = z.permute([2, 1, 0])
        mm_D_N = torch.einsum('ijk,ijk->ijk', [mm_L_D, zz]).sum(dim=0)
        kk_D_ii = torch.einsum('ijk,ijk->ijk', [kk_L_D_ii, zz]).sum(dim=0)
        kk_D_NN = []
        for d in range(self.D):
            kk_D_NN.append(torch.diag(kk_D_ii[d]))
        kk_D_NN = torch.stack(kk_D_NN, dim=0)

        F = pyro.sample('F', dist.MultivariateNormal(mm_D_N, kk_D_NN))
        Y = pyro.sample('Y',
                        dist.MultivariateNormal(F.t(), torch.diag(1 / beta)),
                        obs=self.Y)

        print('************************************************')
        print("v.shape", v.shape)
        print("z.shape", z.shape)
        print("u.shape", u.shape)
        print("X.shape", X.shape)
        print('F.shape', F.shape)
        print('Y.shape', Y.shape)
        print('************************************************')

        return {'X': X, 'u': u, 'F': F, 'v': v, 'z': z}


# -------------------------------------------------------------------------
# -------------------------------test code---------------------------------
# -------------------------------------------------------------------------

from scipy.io import loadmat
import yaml

with open('./configs/toy1.yml', 'rt', encoding='UTF-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

device = torch.device(config['device'])
datatype = torch.float64

data = loadmat("./datasets/toy1.mat")
Y = torch.from_numpy(data["Y"]).to(datatype).to(device)
X = loadmat("./datasets/tt_toy1_X.mat")
X = torch.from_numpy(X["tt_X"]).to(datatype).to(device)
Xu = loadmat("./datasets/tt_toy1_Xu.mat")["tt_Xu"][0]
Xu = torch.stack([torch.from_numpy(Xu[i]) for i in range(len(Xu))], dim=0)

# print("==================The model of MSGPLVM is :=======================")
msgplvm = MSGPLVM(config=config, Y=Y, X=X)
pyro.render_model(model=msgplvm.model, render_distributions=True)
# model = msgplvm.model()
# msgplvm.test_para()