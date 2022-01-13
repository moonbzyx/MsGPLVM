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
from pyro.optim import DCTAdam
import pyro.ops.stats as stats
from pyro.distributions.util import eye_like
from torch.distributions import transform_to
from sklearn.datasets import load_iris
import pandas as pd

torch.set_default_dtype(torch.float64)
data = load_iris()
# print(dir(data))  # 查看data所具有的属性或方法
# print(data.DESCR)  # 查看数据集的简介
y = torch.tensor(data.data, dtype=torch.float64).t()
#直接读到pandas的数据框中
# pd.DataFrame(data=data.data, columns=data.feature_names)
# With y as the 2D Iris data of shape 150x4 and we want to reduce its dimension
# to a tensor X of shape 150x2, we will use GPLVM.
# First, define the initial values for X parameter:
X_init = torch.zeros(150, 2)
# Then, define a Gaussian Process model with input X_init and output y:
kernel = gp.kernels.RBF(input_dim=2, lengthscale=torch.ones(2))
Xu = torch.zeros(20, 2)  # initial inducing inputs of sparse model
gpmodule = gp.models.SparseGPRegression(X_init, y, kernel, Xu)
# Finally, wrap gpmodule by GPLVM, optimize, and get the "learned" mean of X:
gplvm = gp.models.GPLVM(gpmodule)
gp.util.train(gplvm)
X = gplvm.X