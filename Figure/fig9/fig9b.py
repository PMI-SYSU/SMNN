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
hidden_shape=100
P=3
model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P)
_=model.load_state_dict(torch.load(save_path+'model/H_{}_P_{}_{}.pth'.format(hidden_shape,P,0)))
_=model.to(device)
W=torch.matmul(model.l*model.pin,model.pout.T).to(device)
k_range=np.arange(0,W.numel()+W.numel()//10,W.numel()//10)
pruning_rate=[]
R2=[]
for k in k_range:
    Wr=W
    if k==0:
        value=[torch.abs(Wr).min()]
    else:
        value,indice=torch.topk(-torch.abs(Wr).flatten(),k)
    mask=torch.where(torch.abs(Wr)>-value[-1],1,0)
    Wr=Wr*mask
    pr=1-Wr.count_nonzero()/Wr.numel()
    pruning_rate.append(pr.cpu().numpy())
    print("pruning rate:",pruning_rate[-1])
    Win=model.Win.to(device)
    Wout=model.Wout.to(device)
    model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P,mdl=0)
    _=model.to(device)
    model.Win=Win
    model.Wout=Wout
    model.Wr=torch.nn.Parameter(Wr)
    r=test_r2(model)
    print("R2:",np.mean(r))
    R2.append(r)
pruning_rate=np.array(pruning_rate)
prune_R2=np.array(R2)
fig, ax = plt.subplots(1, 1, figsize=(8, 5),dpi=200)
mean=np.array([np.mean(x) for x in prune_R2])
std=np.array([np.std(x) for x in prune_R2])
yerr = np.zeros([2,len(mean)])
yerr[0,:] = std
yerr[1,:] = std
ax.errorbar(pruning_rate*100,mean,yerr=yerr[:,:],ecolor='k',elinewidth=0.5,marker='.',\
    mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--")
plt.ylabel('MSE',fontsize=30)
plt.xlabel('Pruning rate(%)',fontsize=30)