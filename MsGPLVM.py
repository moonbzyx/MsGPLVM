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
from utils.statistics import Psi


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
        self.device = torch.device(self.device)

        self.setup_model()

    def setup_model(self):
        if self.X == None:
            self.X = ppca(self.Y, self.Q)
        self.X = self.X * 1e-7
        # print(self.X[0, :4])
        # plt.plot(self.X)
        # plt.show()

        self.to(self.device)
        # self.Xu = init_inducing(self.init_X, self.M, self.L)  # M * Q
        xu = []  # L*M*Q
        for l in range(self.L):
            xu.append(stats.resample(self.X.clone(), self.M))
        xu = torch.stack(xu, dim=0)

        # def the 'model' parameters as global variables
        self.alpha = pyro.param("alpha",
                                lambda: torch.rand(self.L, device=self.device))
        self.beta = pyro.param("beta",
                               lambda: torch.rand(self.D, device=self.device))
        self.sigma = pyro.param("sigma",
                                lambda: torch.rand(self.L, device=self.device))
        self.w = pyro.param(
            "w", lambda: torch.rand(self.L, self.Q, device=self.device))

        # def the inducing points Xu as global parameters
        self.Xu = pyro.param("Xu", xu)
        # print('self.Xu.shape', self.Xu.shape)

    @pyro_method
    def model(self):
        self.set_mode('model')

        # prepare the plates of model
        p_N = pyro.plate('N', self.N, device=self.device)  # num of instance
        # p_M = pyro.plate('M', self.M, device=self.device)  # num of inducing points
        p_D = pyro.plate('D', self.D,
                         device=self.device)  # dim of output(observation)
        # p_Q = pyro.plate('Q', self.Q, device=self.device)  # dim of input(latent)
        # p_L = pyro.plate('L', self.L, device=self.device)  # num of GP components

        # X is different from self.X (ppca from Y)
        X = pyro.sample(
            'X',
            dist.MultivariateNormal(
                torch.zeros(self.N, self.Q, device=self.device),
                torch.eye(self.Q, device=self.device)))

        # sampling v and z
        with p_D:
            v = pyro.sample('v', dist.Dirichlet(self.alpha))
            with p_N:
                z = pyro.sample('z', dist.Multinomial(1, v))

        # calculate parts of the kerneal groups
        k_L_MM = []  # kernels : Xu_Xu-->u
        k_L_NN = []  # kernels : X_X
        k_L_NM = []  # kernels : X_Xu
        k_L_MN = []  # kernels : Xu_X
        for l in range(self.L):
            k = RBFard(input_dim=self.Q,
                       variance=self.sigma[l],
                       lengthscale=self.w[l])
            k_MM = k(self.Xu[l])  # k_MM used to add jitter
            k_MM.view(-1)[::self.M + 1] += self.jitter
            k_L_MM.append(k_MM)
            k_L_NN.append(k(X))
            k_L_NM.append(k(X, self.Xu[l]))
            k_L_MN.append(k(self.Xu[l], X))
        k_L_MM = torch.stack(k_L_MM, dim=0)
        k_L_NN = torch.stack(k_L_NN, dim=0)
        k_L_NM = torch.stack(k_L_NM, dim=0)
        k_L_MN = torch.stack(k_L_MN, dim=0)

        # the inducing outputs, L*D
        # the number of it's feature is M --> final shape: D*L*M
        u = pyro.sample(
            'u',
            dist.MultivariateNormal(
                torch.zeros(self.D, self.L, self.M, device=self.device),
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

        # mean field theory
        kk_L_D_ii = []
        for l in range(self.L):
            kk_D_ii = torch.diag(kk_L[l]).unsqueeze(0).expand([self.D, self.N])
            kk_L_D_ii.append(kk_D_ii)
        kk_L_D_ii = torch.stack(kk_L_D_ii, dim=0)

        # select GPs (for L kernels) for every pixel of data
        zz = z.permute([2, 1, 0])
        mm_D_N = torch.einsum('ijk,ijk->ijk', [mm_L_D, zz]).sum(dim=0)
        kk_D_ii = torch.einsum('ijk,ijk->ijk', [kk_L_D_ii, zz]).sum(dim=0)
        kk_D_NN = []
        for d in range(self.D):
            kk_D_NN.append(torch.diag(kk_D_ii[d]))
        kk_D_NN = torch.stack(kk_D_NN, dim=0)

        # sampling F and Y
        F = pyro.sample('F', dist.MultivariateNormal(mm_D_N, kk_D_NN))
        Y = pyro.sample('Y',
                        dist.MultivariateNormal(F.t(),
                                                torch.diag(1 / self.beta)),
                        obs=self.Y)

        return {'X': X, 'u': u, 'F': F, 'v': v, 'z': z}

    @pyro_method
    def guide(self):
        self.set_mode('guide')

        # initialize the parameters
        mu = pyro.param("mu", self.X.t())  # Q * N
        S = pyro.param(
            "S",
            torch.eye(self.N, device=self.device).expand(
                [self.Q, self.N, self.N]) * 0.1)

        # phi is related to the Dirichlet distribution
        # so sum of phi_id shuould be equal to 1
        phi = pyro.param("phi",
                         torch.rand(self.N, self.D, self.L,
                                    device=self.device),
                         constraint=constraints.simplex)

        gamma = pyro.param("gamma",
                           torch.rand(self.D, self.L, device=self.device))
        # Xu = pyro.param("Xu", self.Xu)  # it's not parameter in 'model'

        v = pyro.sample('v', dist.Dirichlet(gamma))
        z = pyro.sample('z', dist.Multinomial(1, phi))

        # X has the parameters: mu, S
        X = pyro.sample('X', dist.MultivariateNormal(mu, S))
        # print(X.device)
        X = X.reshape(self.N, self.Q)  # aligned with the 'X' in 'model'

        # calculate parts of the kerneal groups
        k_L_MM = []  # kernels : Xu_Xu-->u
        k_L_NN = []  # kernels : X_X
        k_L_NM = []  # kernels : X_Xu
        k_L_MN = []  # kernels : Xu_X
        for l in range(self.L):
            k = RBFard(input_dim=self.Q,
                       variance=self.sigma[l],
                       lengthscale=self.w[l])
            k_MM = k(self.Xu[l])  # k_MM used to add jitter
            k_MM.view(-1)[::self.M + 1] += self.jitter
            k_L_MM.append(k_MM)
            k_L_NN.append(k(X))
            k_L_NM.append(k(X, self.Xu[l]))
            k_L_MN.append(k(self.Xu[l], X))
        k_L_MM = torch.stack(k_L_MM, dim=0)
        k_L_NN = torch.stack(k_L_NN, dim=0)
        k_L_NM = torch.stack(k_L_NM, dim=0)
        k_L_MN = torch.stack(k_L_MN, dim=0)

        # calculate the mean and kernel of  variational distribution q(u)
        psi = Psi(phi, self.sigma, self.w, mu, S, self.Xu, self.device)
        psi_1 = psi.psi_1()
        psi_2 = psi.psi_2_loop()
        m_L_D_M = []
        k_L_D_MM = []
        for l in range(self.L):
            mm_D = []
            kk_D = []
            for d in range(self.D):
                tem = k_L_MM[l] @ torch.linalg.inv(self.beta[d] * psi_2[l, d] +
                                                   k_L_MM[l])
                # + torch.eye(self.M) * self.jitter
                mm = self.beta[d] * tem @ (psi_1[l, d]).t() @ self.Y[:, d]
                kk = tem @ k_L_MM[l]
                mm_D.append(mm)
                kk_D.append(kk)
            mm_D = torch.stack(mm_D, dim=0)
            kk_D = torch.stack(kk_D, dim=0)
            m_L_D_M.append(mm_D)
            k_L_D_MM.append(kk_D)
        m_L_D_M = torch.stack(m_L_D_M, dim=0)
        k_L_D_MM = torch.stack(k_L_D_MM, dim=0)

        # sampling u
        u = pyro.sample('u',
                        dist.MultivariateNormal(m_L_D_M, k_L_D_MM).to_event(1))

        # calculate the mean and kernel of 'f ~ GPs'
        u = u.permute([1, 0, 2])
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

        # mean field theory
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

        # sampling F
        F = pyro.sample('F', dist.MultivariateNormal(mm_D_N, kk_D_NN))

        return {'X': X, 'u': u, 'F': F, 'v': v, 'z': z}


# -------------------------------------------------------------------------
# -------------------------------test code---------------------------------
# -------------------------------------------------------------------------

# from scipy.io import loadmat
# import yaml

# with open('./configs/toy1.yml', 'rt', encoding='UTF-8') as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)

# device = torch.device(config['device'])
# datatype = torch.float64

# data = loadmat("./datasets/toy1.mat")
# Y = torch.from_numpy(data["Y"]).to(datatype).to(device)
# X = loadmat("./datasets/tt_toy1_X.mat")
# X = torch.from_numpy(X["tt_X"]).to(datatype).to(device)
# Xu = loadmat("./datasets/tt_toy1_Xu.mat")["tt_Xu"][0]
# Xu = torch.stack([torch.from_numpy(Xu[i]) for i in range(len(Xu))], dim=0)

# msgplvm = MSGPLVM(config=config, Y=Y, X=X)
# msgplvm.model()
# # pyro.render_model(model=msgplvm.model, render_distributions=True)
# pyro.render_model(model=msgplvm.guide, render_distributions=True)
# print('*' * 15, 'MsGPLVM class parameters', '*' * 15)
# print(list(pyro.get_param_store().keys()))
