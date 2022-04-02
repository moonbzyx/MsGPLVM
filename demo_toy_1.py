"""
MODEL: THE MSGPLVM MODEL
COPYRIGHT: CHAOJIEMEN 2022-1-4
"""

from re import T
from scipy.io import loadmat
import matplotlib.pyplot as plt
import datetime
import yaml
import torch
from utils.ppca import ppca
from MsGPLVM import *
import pyro

import os
import torch.distributions.constraints as constraints
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, TraceGraph_ELBO
from utils.utils import *
import sys

from tqdm import tqdm
from utils.log_tqdm import my_log
# assert pyro.__version__.startswith('1.8.0')

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
max_steps = 2 if smoke_test else 10000

# informations of the experiment
TIME = datetime.datetime.now()
DATE = TIME.strftime("%Y-%m-%d")
TT = TIME.strftime("%H:%M:%S")
EXPERIMENT_NO = 3

seed = 6
manual_seed(seed)

# open the configures
with open('./configs/toy1.yml', 'rt', encoding='UTF-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

log = my_log(f'./results/msgplmv_exp_{EXPERIMENT_NO}')
log.info(f'-----The begin time: {DATE}_{TT}-----')

# settings of data and device
torch.set_num_threads(4)
device = torch.device(config['device'])
datatype = torch.float64

log.info(f'The device is : {device}')

# prepare the data
data = loadmat("./datasets/toy1.mat")
Y = torch.from_numpy(data["Y"]).to(datatype).to(device)
X = loadmat("./datasets/tt_toy1_X.mat")
X = torch.from_numpy(X["tt_X"]).to(datatype).to(device)
Xu = loadmat("./datasets/tt_toy1_Xu.mat")["tt_Xu"][0]
Xu = torch.stack([torch.from_numpy(Xu[i]) for i in range(len(Xu))], dim=0)
Xu = Xu.to(device)

# build the model
# msgplvm = MSGPLVM(config, Y, X)
msgplvm = MSGPLVM2(config, Y, X)
msgplvm.to(device)
# print(list(pyro.get_param_store().keys()))

# clear the param store in case we're in a REPL
# pyro.clear_param_store()
# print(list(pyro.get_param_store().keys()))

# setup the optimizer and the inference algorithm
num_step = 300
lr = 0.0015
# gamma = 0.1
# optimizer = optim.Adam({"lr": lr, "betas": (0.93, 0.999)})
optimizer = optim.Adam({"lr": lr, "betas": (0.95, 0.999)})
# lrd = gamma**(1. / num_step)
# optimizer = optim.ClippedAdam({'lr': lr, 'lrd': lrd})
svi = SVI(msgplvm.model,
          msgplvm.guide,
          optimizer,
          loss=pyro.infer.Trace_ELBO(retain_graph=True))

# print(msgplvm.get_param("gamma"))
# training or doing inference
# a = pyro.param("w")
# print(a)
i_tqdm = tqdm(range(num_step))
for i in i_tqdm:

    loss = svi.step()

    if i % 30 == 0:
        log.info(f'Loss is : {loss}')
    if i == 0:
        test_begin = msgplvm.get_param("w")
        log.info(f'The w before training is : \n {test_begin}')
    # if i == num_step - 1 or loss <= 0.0:
    if i == num_step - 1:
        test_end = msgplvm.get_param("w")
        log.info(f'Loss is : {loss}')
        log.info(f'The w after training is : \n {test_end}')
        # if loss <= 0.0:
        #     break
    i_tqdm.set_description(f'loss is {loss}')
    # print('\n', msgplvm.get_param("w"))

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
log.trace(f'-----The end time: {DATE}_{TT}-----')
