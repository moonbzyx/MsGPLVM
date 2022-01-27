import torch
from torch.distributions import constraints

from pyro.contrib.gp.kernels.kernel import Kernel
from pyro.nn.module import PyroParam
from loguru import logger


def _torch_sqrt(x, eps=1e-12):
    """
    A convenient function to avoid the NaN gradient issue of :func:`torch.sqrt`
    at 0.
    """
    return (x + eps).sqrt()


class Isotropy(Kernel):
    """
    Base class for a family of isotropic covariance kernels which are functions of the
    distance :math:`|x-z|/l`, where :math:`l` is the length-scale parameter.

    By default, the parameter ``lengthscale`` has size 1. To use the isotropic version
    (different lengthscale for each dimension), make sure that ``lengthscale`` has size
    equal to ``input_dim``.

    :param torch.Tensor lengthscale: Length-scale parameter of this kernel.
    """
    def __init__(self,
                 input_dim,
                 variance=None,
                 lengthscale=None,
                 active_dims=None):
        super().__init__(input_dim, active_dims)

        self.variance = torch.tensor(1.0) if variance is None else variance
        # self.variance = PyroParam(self.variance, constraints.positive)

        self.lengthscale = torch.tensor(
            1.0) if lengthscale is None else lengthscale
        # self.lengthscale = PyroParam(self.lengthscale, constraints.positive)

    # @logger.catch
    def _square_scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`l * \|\frac{X-Z}\|^2`.
        """

        if X.size(1) != self.input_dim:
            raise ValueError(f"Input features and input_dim should be equal.\
                inputs is {self.input_dim}, features is {X.size(1)}")
        if Z is None:
            Z = X
        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        ll = _torch_sqrt(self.lengthscale)

        if self.lengthscale.numel() == 1:
            ll = torch.eye(self.input_dim, device=X.device) * ll
        elif self.lengthscale.size(0) == self.input_dim:
            ll = torch.diag(ll)
        else:
            raise ValueError(
                "Lengthscale size and input features should be equal")

        scaled_X = X @ ll
        scaled_Z = Z @ ll
        X2 = (scaled_X**2).sum(1, keepdim=True)
        Z2 = (scaled_Z**2).sum(1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.t())
        r2 = X2 - 2 * XZ + Z2.t()
        return r2.clamp(min=0)

    def _scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """
        return _torch_sqrt(self._square_scaled_dist(X, Z))

    def _diag(self, X):
        """
        Calculates the diagonal part of covariance matrix on active features.
        """
        return self.variance.expand(X.size(0))


class RBFard(Isotropy):
    r"""
    Implementation of Radial Basis Function kernel:

        :math:`k(x,z) = \sigma^2\exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

    .. note:: This kernel also has name `Squared Exponential` in literature.
    """
    def __init__(self,
                 input_dim,
                 variance=None,
                 lengthscale=None,
                 active_dims=None):
        super().__init__(input_dim, variance, lengthscale, active_dims)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        r2 = self._square_scaled_dist(X, Z)
        return self.variance**2 * torch.exp(-0.5 * r2)


if __name__ == '__main__':

    X = torch.tensor([[1.0, 1], [1, 1]])
    # X = torch.tensor([[0.0, 2, 3], [3, 4, 4], [5, 6, 9]])
    X_sqrt = X.sqrt()
    X2 = (X**2).sum(1, keepdim=True)
    l = torch.tensor([1, 3])
    # k = RBFard(input_dim=2, lengthscale=l)
    k = RBFard(input_dim=2)

    print(k(X))
    print(k.lengthscale)
    print(k.variance)
