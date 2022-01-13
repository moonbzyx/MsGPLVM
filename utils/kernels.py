import pyro.contrib.gp.kernels
from pyro.contrib.gp.kernels import Kernel
from torch.distributions import constraints
from pyro.contrib import gp
import torch
from pyro.nn.module import PyroParam

torch.set_default_dtype(torch.float64)


class Linard2(Kernel):
    r"""
    Implementation of Linear ARD2 Kernel
    """
    def __init__(self,
                 input_dim,
                 input_scale=None,
                 active_dims=None,
                 *arg,
                 **kwargs):
        super().__init__(input_dim, active_dims)

        input_scale = torch.ones(
            [input_dim], dtype=torch.float64
        ) if input_scale is None else input_scale * torch.ones([input_dim])
        self.input_scale = PyroParam(input_scale,
                                     constraint=constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if Z is None:
            Z = X
        X = self._slice_input(X)
        if diag:
            pass
        Z = self._slice_input(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        return X.matmul(torch.diag(self.input_scale).matmul(Z.t()))


class Rbfard2(Kernel):
    r"""
    Implementation of RBF ARD2 Kernel
    """
    def __init__(self,
                 input_dim,
                 variance=1,
                 inputScales=0.999,
                 active_dims=None,
                 *arg,
                 **kwargs):
        super().__init__(input_dim, active_dims)
        self.variance = PyroParam(torch.tensor(variance), constraints.positive)
        inputScales = torch.tensor(inputScales)
        inputScales = inputScales * torch.ones(input_dim,
                                               device=inputScales.device)
        self.inputScales = PyroParam(inputScales, constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if Z is None:
            Z = X
        X = self._slice_input(X)
        if diag:
            pass
        Z = self._slice_input(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")
        scales = torch.diag(self.inputScales.sqrt())
        print(scales)
        print("X is :\n", X)
        # X = torch.matmul(X, scales)
        # Z = torch.matmul(Z, scales)
        X = X @ scales
        Z = Z @ scales
        X2 = (X**2).sum(1, keepdim=True)
        Z2 = (Z**2).sum(1, keepdim=True)
        XZ = X.matmul(Z.t())
        r2 = X2 - 2 * XZ + Z2.t()
        r2 = r2.clamp(min=0)
        return self.variance * torch.exp(r2 * -0.5)


class Rbf_plus(Kernel):
    def __init__(self, input_dim, active_dims=None, *arg, **kwargs):
        super().__init__(input_dim, active_dims)
        self.rbf = gp.kernels.RBF(input_dim=input_dim)
        # self.whitenoise = gp.kernels.WhiteNoise(input_dim=input_dim, variance=torch.tensor(-2).exp())
        self.whitenoise = gp.kernels.WhiteNoise(input_dim=input_dim,
                                                variance=torch.tensor(1e-2))
        self.bias = gp.kernels.Constant(input_dim=input_dim,
                                        variance=torch.tensor(0.1))

    def forward(self, X, Z=None, diag=False):
        return self.rbf(X, Z, diag) + self.whitenoise(X, Z, diag) + self.bias(
            X, Z, diag)


class Linard2_plus(Kernel):
    def __init__(self, input_dim, active_dims=None, *arg, **kwargs):
        super().__init__(input_dim, active_dims)
        self.linard2 = Linard2(input_dim=input_dim)
        self.whitenoise = gp.kernels.WhiteNoise(
            input_dim=input_dim, variance=torch.tensor(-2).exp())
        self.bias = gp.kernels.Constant(input_dim=input_dim,
                                        variance=torch.tensor(0.1))

    def forward(self, X, Z=None, diag=False):
        return self.linard2(X, Z, diag) + self.whitenoise(
            X, Z, diag) + self.bias(X, Z, diag)


class Rbfard2_plus(Kernel):
    def __init__(self,
                 input_dim,
                 variance=1,
                 inputScales=1,
                 active_dims=None,
                 *arg,
                 **kwargs):
        super().__init__(input_dim, active_dims)
        self.rbfard2 = Rbfard2(input_dim=input_dim,
                               variance=variance,
                               inputScales=inputScales)
        self.whitenoise = gp.kernels.WhiteNoise(
            input_dim=input_dim, variance=torch.tensor(-2).exp())
        self.bias = gp.kernels.Constant(input_dim=input_dim,
                                        variance=torch.tensor(-2).exp())

    def forward(self, X, Z=None, diag=False):
        return self.rbfard2(X, Z, diag) + self.whitenoise(
            X, Z, diag) + self.bias(X, Z, diag)


class Eye(Kernel):
    def __init__(self, input_dim, activate_dims=None, *arg, **kwargs):
        super().__init__(input_dim, activate_dims)

    def forward(self, X, Z=None, diag=False):
        N = X.shape[0]
        return torch.eye(N, N, device=X.device)


mapping = {
    'rbf_plus': Rbf_plus,
    'linard2_plus': Linard2_plus,
    'rbfard2_plus': Rbfard2_plus,
    'rbf': gp.kernels.RBF,
    'eye': Eye,
}


def get_kernel_obj(name='rbf_plus'):
    assert name in mapping.keys()
    return mapping[name]


if __name__ == '__main__':

    X = torch.tensor([[1.0, 2], [3, 4], [5, 6]])
    X2 = (X**2).sum(1, keepdim=True)
    # k = Rbfard2(input_dim=2, lengthscale=torch.tensor([1, 2]))
    k = Rbfard2(input_dim=2)

    print(k(X))
    # print(X2)