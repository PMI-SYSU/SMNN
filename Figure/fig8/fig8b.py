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
tref0_path='Results/mante/tref0/'#save path for SMNN with tref=0
replace_path='Results/mante/replace/'#save path for SMNN with approximate step function

P_range=[1,2,3,5,30,100]
N_range=[100]
mante_R=[]
tref0_R=[]
replace_R=[]

R=[]
for hidden_shape in N_range:
    R2=[]
    for P in P_range:
        r=[]
        for i in range(3):
            a=torch.load(replace_path+'/train_loss/H_{}_P_{}_{}.pth'.format(hidden_shape,P,i))
            if np.isnan(a[-1]):
                continue
            model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P,replace=1)
            _=model.load_state_dict(torch.load(replace_path+'model/H_{}_P_{}_{}.pth'.format(hidden_shape,P,i)))
            _=model.to(device)
            r.append(test_r2(model))
        R2.append(np.concatenate(r))
    R2=np.array(R2)
    R.append(R2)
replace_R=R

R=[]
for hidden_shape in N_range:
    R2=[]
    for P in P_range:
        r=[]
        for i in range(3):
            model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P,'double')
            _=model.load_state_dict(torch.load(save_path+'model/H_{}_P_{}_{}.pth'.format(hidden_shape,P,i)))
            _=model.to(device)
            r.append(test_r2(model))
        R2.append(np.concatenate(r))
    R2=np.array(R2)
    R.append(R2)
mante_R=R

R=[]
for hidden_shape in N_range:
    R2=[]
    for P in P_range:
        r=[]
        for i in range(3):
            model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P,tref=0)
            _=model.load_state_dict(torch.load(tref0_path+'model/H_{}_P_{}_{}.pth'.format(hidden_shape,P,i)))
            _=model.to(device)
            r.append(test_r2(model))
        R2.append(np.concatenate(r))
    R2=np.array(R2)
    R.append(R2)
tref0_R=R


fig, ax = plt.subplots(1, 1, figsize=(8, 5),dpi=200)

i=0
for R2 in mante_R:
    R2_mean=np.array([np.mean(x) for x in R2])
    R2_std=np.array([np.std(x) for x in R2])
    yerr = np.zeros([2,len(R2_mean)])
    yerr[0,:] = R2_std
    yerr[1,:] = R2_std
    ax.errorbar(P_range,R2_mean,yerr=yerr[:,:],ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='SMNN')
    i+=1


i=0
for R2 in tref0_R:
    R2_mean=np.array([np.mean(x) for x in R2])
    R2_std=np.array([np.std(x) for x in R2])
    yerr = np.zeros([2,len(R2_mean)])
    yerr[0,:] = R2_std
    yerr[1,:] = R2_std
    ax.errorbar(P_range,R2_mean,yerr=yerr[:,:],ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label=r'SMNN $t_{ref}$=0')
    i+=1

i=0
for R2 in replace_R:
    R2_mean=np.array([np.mean(x) for x in R2])
    R2_std=np.array([np.std(x) for x in R2])
    yerr = np.zeros([2,len(R2_mean)])
    yerr[0,:] = R2_std
    yerr[1,:] = R2_std
    ax.errorbar(P_range,R2_mean,yerr=yerr[:,:],ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='SMNN approx.')
    i+=1


plt.legend(loc=1,ncol=1,fontsize=15,columnspacing=0.4)
plt.xscale('log')
plt.ylabel('MSE',fontsize=30)
plt.xlabel('Mode Size ($P$)',fontsize=30)