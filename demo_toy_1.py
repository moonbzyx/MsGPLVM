"""
MODEL: THE MSGPLVM MODEL
COPYRIGHT: CHAOJIEMEN 2022-1-4
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import datetime
import yaml
import torch
from utils.ppca import ppca
from MsGPLVM import MSGPLVM
import pyro

import os
import torch.distributions.constraints as constraints
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, TraceGraph_ELBO
import sys

from tqdm import tqdm
from utils.log_tqdm import my_log
assert pyro.__version__.startswith('1.8.0')

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
max_steps = 2 if smoke_test else 10000

# informations of the experiment
TIME = datetime.datetime.now()
DATE = TIME.strftime("%Y-%m-%d")
TT = TIME.strftime("%H:%M:%S")
EXPERIMENT_NO = 1
# open the configures
with open('./configs/toy1.yml', 'rt', encoding='UTF-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# settings of data and device
torch.set_num_threads(4)
device = torch.device(config['device'])
datatype = torch.float64

# prepare the data
data = loadmat("./datasets/toy1.mat")
Y = torch.from_numpy(data["Y"]).to(datatype).to(device)
X = loadmat("./datasets/tt_toy1_X.mat")
X = torch.from_numpy(X["tt_X"]).to(datatype).to(device)
Xu = loadmat("./datasets/tt_toy1_Xu.mat")["tt_Xu"][0]
Xu = torch.stack([torch.from_numpy(Xu[i]) for i in range(len(Xu))], dim=0)
Xu = Xu.to(device)

# build the model
msgplvm = MSGPLVM(config, Y, X).to(device)

# clear the param store in case we're in a REPL
pyro.clear_param_store()

# setup the optimizer and the inference algorithm
lr = 0.01
optimizer = optim.Adam({"lr": lr, "betas": (0.93, 0.999)})
svi = SVI(msgplvm.model, msgplvm.guide, optimizer, loss=TraceGraph_ELBO())

# training or doing inference
log = my_log(f'./results/msgplmv_exp{EXPERIMENT_NO}')
i_tqdm = tqdm(range(3000))
for i in i_tqdm:
    loss = svi.step()
    if i % 30 == 0:
        log.info(f'Loss is : {loss}')
    i_tqdm.set_description(f'loss is {loss}')

# for k in range(self.max_steps):
#     svi.step(use_decaying_avg_baseline)
#     if k % 100 == 0:
#         print('.', end='')
#         sys.stdout.flush()

# pyro.render_model(model=msgplvm.model, render_distributions=True)
# pyro.render_model(model=msgplvm.guide, render_distributions=True)

# print(Xu.device)
# plt.plot(Xu[0])
# plt.show()
