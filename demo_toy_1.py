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

TIME = datetime.datetime.now()
DATE = TIME.strftime("%Y-%m-%d")
TT = TIME.strftime("%H:%M:%S")
EXPERIMENT_NO = 1

with open('./configs/toy1.yml', 'rt', encoding='UTF-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

torch.set_num_threads(4)
device = torch.device(config['device'])
datatype = torch.float64

data = loadmat("./datasets/toy1.mat")
Y = torch.from_numpy(data["Y"]).to(datatype).to(device)
X = loadmat("./datasets/tt_toy1_X.mat")
X = torch.from_numpy(X["tt_X"]).to(datatype).to(device)
Xu = loadmat("./datasets/tt_toy1_Xu.mat")["tt_Xu"][0]
Xu = torch.stack([torch.from_numpy(Xu[i]) for i in range(len(Xu))], dim=0)
print(Xu.shape)
# plt.plot(Xu)
# plt.show()
