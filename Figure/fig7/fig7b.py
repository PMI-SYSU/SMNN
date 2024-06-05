import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {
         'figure.figsize': ((15,8)),
         'legend.fontsize': 'x-large',
         'xtick.labelsize':20,
         'ytick.labelsize':20,
         'axes.labelsize': 'xx-large',
         'axes.spines.top': True,
         'axes.spines.right': True,
         'axes.titlesize': 'xx-large',}
pylab.rcParams.update(params)
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 200 
import numpy as np
from SMNN.Figure.mante_utils import *
from SMNN.mante.model import *
import torch
save_path='Results/mante/' #save path for SMNN
hidden_shape=200
P=100
model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P,'double')
_=model.load_state_dict(torch.load(save_path+'model/H_{}_P_{}_0.pth'.format(hidden_shape,P)))
_=model.to(device)
spike_times=[]
for i in range(300):
    input_data,targets=generate_inputs()
    with torch.no_grad():
        o=model(input_data).flatten(0)
    spk=model.S.flatten(1).cpu()
    spike_times.append(spk.sum(0))
spike_times=torch.stack(spike_times,0)
fano_factor=spike_times.var(0)/spike_times.mean(0)
plt.figure(figsize=(8,5),dpi=200)
plt.hist(fano_factor.numpy(),bins=30,edgecolor='k',alpha=0.7)
plt.xlabel('Fano factor',fontsize=30)
plt.ylabel('Number of cells',fontsize=30)