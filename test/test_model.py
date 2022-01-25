import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import os

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.0')

# alpha = 1
# beta = 10
# Q: 8
# D: 30
# L: 3
# M: 50
# N: 100


def model():
    alpha = torch.rand((1, 3))
    X = pyro.sample('X', dist.MultivariateNormal(torch.zeros(8), torch.eye(8)))
    with pyro.plate("D", 30):
        v = pyro.sample('v', dist.Dirichlet(alpha))
        with pyro.plate("N", 100):
            z = pyro.sample('z', dist.Multinomial(v.shape[1], v))
    print("X.shape is :", X.shape)
    print("v.shape is :", v.shape)
    print("z.shape is :", z.shape)
    pass


class A():
    data = 1

    def model(self, x):
        print("x is :\n", x)
        alpha = torch.rand((1, 3))
        X = pyro.sample('X',
                        dist.MultivariateNormal(torch.zeros(8), torch.eye(8)))
        with pyro.plate("D", 30):
            v = pyro.sample('v', dist.Dirichlet(alpha))
            with pyro.plate("N", 100):
                z = pyro.sample('z', dist.Multinomial(v.shape[1], v))
        print("X.shape is :", X.shape)
        print("v.shape is :", v.shape)
        print("z.shape is :", z.shape)
        pass

    # def v_model(self):
    #     pyro.render_model(model=self.model)


a = A()
# a.v_model()
# mod = a.model
pyro.render_model(model=a.model, model_args=(3, ))

# a = A()
# a.v_model()

# data = torch.ones(10)
# model(data)
# pyro.render_model(a.model)
# pyro.render_model(model)

# alpha = torch.rand((1, 3))
# alpha = pyro.param('alpha', lambda: torch.rand(1,3))
# print(alpha)
# pyro.param('test', lambda: torch.rand(1,3))
# alpha = pyro.param('alpha', lambda: torch.rand(1,3))
# beta = 10 * torch.rand((1, 30))
# X = pyro.sample('X', dist.Normal(0, 1))
# v = pyro.sample('v', dist.Dirichlet(alpha))
# with pyro.plate("D",30)
# with pyro.plate("N", len(data)):
#     pyro.sample("obs", dist.Normal(0, 1), obs=data)