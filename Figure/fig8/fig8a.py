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
from SMNN.Figure.mnist_utils import *
from SMNN.mnist.model import *
import torch
save_path='Results/mnist/' #save path for SMNN
tref0_path='Results/mnist/tref0/'#save path for SMNN with tref=0
replace_path='Results/mnist/replace/'#save path for SMNN with approximate step function
P_range=[1,2,3,5,30,100]
N_range=[100]
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot()
name='acc'
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
ax=plt.gca()
axins = ax.inset_axes((0.6, 0.15, 0.35, 0.3))
j=0
for hidden_shape in N_range:
    results=[]
    for P in P_range:
        acc=[]
        for i in range(3):
            a = torch.load(save_path+'{}/H_{}_P_{}_{}.pth'.format(name,hidden_shape,P,i))
            acc.append(a[-1]*100)
        results.append(acc)
    results=np.array(results)
    mean=np.array([np.mean(x) for x in results])
    std=np.array([np.std(x) for x in results])
    yerr = np.zeros([2,len(P_range)])
    yerr[0,:] = std
    yerr[1,:] = std
    ax.errorbar(P_range,mean,yerr=yerr[:,:],ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='SMNN')
    j+=1

j=0
for hidden_shape in N_range:
    results=[]
    for P in P_range:
        acc=[]
        for i in range(3):
            a = torch.load(tref0_path+'{}/H_{}_P_{}_{}.pth'.format(name,hidden_shape,P,i))
            acc.append(a[-1]*100)
        results.append(acc)
    results=np.array(results)
    mean=np.array([np.mean(x) for x in results])
    std=np.array([np.std(x) for x in results])
    yerr = np.zeros([2,len(P_range)])
    yerr[0,:] = std
    yerr[1,:] = std
    ax.errorbar(P_range,mean,yerr=yerr[:,:],ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label=r'SMNN $\tau_{ref}$=0')
    j+=1

j=0
for hidden_shape in N_range:
    results=[]
    for P in P_range:
        acc=[]
        for i in range(3):
            a = torch.load(replace_path+'{}/H_{}_P_{}_{}.pth'.format(name,hidden_shape,P,i))
            acc.append(a[-1]*100)
        results.append(acc)
    results=np.array(results)
    mean=np.array([np.mean(x) for x in results])
    std=np.array([np.std(x) for x in results])
    yerr = np.zeros([2,len(P_range)])
    yerr[0,:] = std
    yerr[1,:] = std
    ax.errorbar(P_range,mean,yerr=yerr[:,:],ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label=r'SMNN aprrox.')
    axins.errorbar(P_range,mean,yerr=yerr[:,:],color='#2ca02c',ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='SMNN approx.')
    j+=1

plt.ylim(94,98)
plt.legend(loc='best',ncol=1,fontsize=15,columnspacing=0.4)
plt.xscale('log')
plt.ylabel('Test acc.(%)',fontsize=30)
plt.xlabel('Mode Size ($P$)',fontsize=30)
axins.set_xscale('log')
axins.set_xticks([1,10,100])
axins.set_ylim(0,22)
axins.set_xlim(0.7,150)