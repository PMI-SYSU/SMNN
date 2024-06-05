import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import snntorch.spikeplot as splt
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
plt.rcParams['savefig.dpi'] = 200 
plt.rcParams['figure.dpi'] = 200 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from SMNN.Figure.mante_utils import *
from SMNN.mante.model import *
from SMNN.Figure.mante_plot import *


save_path='Results/mante/' #save path for SMNN
rate_path='Results/mante/rate/' #save path for MDL RNN
full_path='Results/mante/full/'#save path for SNN
full_rate_path='Results/mante/full_rate/'#save path for RNN
tref0_path='Results/mante/tref0/'#save path for SMNN with tref=0
replace_path='Results/mante/replace/'#save path for SMNN with approximate step function
from sklearn.metrics import mean_squared_error
def test_r2():
    R2=[]
    for i in range(5):
        input_data,targets=generate_inputs()
        with torch.no_grad():
            o=model(input_data).flatten(0).cpu()
        r=r2(targets.flatten(0).cpu(),o)
        R2.append(r)
    return  np.array(R2)
def r2(y_true,y_pred):
    return mean_squared_error(y_true,y_pred)


# Figure 3a
P_range=[1,2,3,5,30,100]
N_range=[100,200]
mante_R=[]
full_R=[]

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
    for P in P_range[2:3]:
        r=[]
        for i in range(3):
            model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P,'double',mdl=0)
            _=model.load_state_dict(torch.load(full_path+'model/H_{}_P_{}_{}.pth'.format(hidden_shape,P,i)))
            _=model.to(device)
            r.append(test_r2(model))
        R2.append(np.concatenate(r))
    R2=np.array(R2)
    R.append(R2)
full_R=R   

fig, ax = plt.subplots(1, 1, figsize=(8, 5),dpi=200)
i=0
for R2 in mante_R:
    R2_mean=np.array([np.mean(x) for x in R2])
    R2_std=np.array([np.std(x) for x in R2])
    yerr = np.zeros([2,len(R2_mean)])
    yerr[0,:] = R2_std
    yerr[1,:] = R2_std
    ax.errorbar(P_range,R2_mean,yerr=yerr[:,:],ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='SMNN $N$={}'.format(N_range[i]))
    i+=1

i=0
for R2 in full_R:
    R2_mean=np.array([np.mean(x) for x in R2])
    R2_std=np.array([np.std(x) for x in R2])
    R2_mean[:]=np.mean(R2_mean)
    R2_std[:]=np.mean(R2_std)
    yerr = np.zeros([2,len(P_range)])
    yerr[0,:] = R2_std
    yerr[1,:] = R2_std
    ax.errorbar(P_range,R2_mean.repeat(len(P_range)),yerr=yerr[:,:],ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='SNN $N$={}'.format(N_range[i]))
    i+=1


plt.legend(loc=1,ncol=2,fontsize=15,columnspacing=0.4)
plt.xscale('log')
plt.ylabel('MSE',fontsize=30)
plt.xlabel('Mode Size ($P$)',fontsize=30)

#Figure 3b
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot()
P_range=[1,2,3,10,50,100]
N_range=[30,50,100,200]
R=[]
name='train_loss'
for P in P_range:
    results=[]
    for N in N_range[2:3]:
        loss=[]
        for i in range(3):
            a=torch.load(save_path+'{}/H_{}_P_{}_{}.pth'.format(name,N,P,i))
            loss.append(a[::50])
        loss=torch.stack(loss,0).float()
        results.append(loss)
    results=torch.tensor(results[0])
    err = np.zeros([2,results.shape[1]])
    mean=results[0]
    std=results[1]

    _=ax.errorbar(np.arange(mean.shape[0])*50,mean,yerr=err[:,:],ecolor='k',elinewidth=0.5,marker='.',\
    mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='$P$={}'.format(P))

plt.legend(loc=1,ncol=2,fontsize=25,columnspacing=0.4)
plt.ylabel('MSE',fontsize=30)
plt.xlabel('Training batch',fontsize=30)

#Figure 3c
for hidden_shape in [100]:
    for P in [3]:
        for i in [0]:
            model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P,'double')
            _=model.load_state_dict(torch.load('Figure/mante/model/H_{}_P_{}_double.pth'.format(hidden_shape,P)))
            _=model.to(device)
test_plot(model)

#Figure 3d
hidden_shape=100
P=3
model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P,'double')
_=model.load_state_dict(torch.load(save_path+'model/H_{}_P_{}_0.pth'.format(hidden_shape,P)))
_=model.to(device)
input_data,targets=generate_inputs()
with torch.no_grad():
    o=model(input_data).flatten(0)
dt=0.2
U=model.U.cpu().detach()
r=model.R.cpu().detach()
r=r.reshape(r.shape[0],-1)
U=U.reshape(U.shape[0],-1)
index=torch.argmax(r,dim=0)
Um=U[index,:].diag()
U=U[:,U[-1,:]>0]
U=U[:,:5]
U=torch.where(U>1,1,U)
fig = plt.figure(facecolor="w", figsize=(12,8),dpi=200)
ax = fig.add_subplot(111)
plt.plot(np.arange(T)*dt,U)
plt.xlabel('Time (ms)',fontsize=30)
plt.ylabel('U',fontsize=30)
plt.fill_between(np.arange(50,250)*dt,torch.min(U)-3,5,color='grey',alpha=0.3)
#plt.scatter(index[[10:13]]*2,Um[[10:13]],c='k',marker='^')
plt.ylim(torch.min(U)-0.1,1.25)
plt.hlines(0,0,T*dt,colors = "k",linewidth=5, linestyles = "dashed")
plt.text(0,-0.1,'$U_{res}$',fontsize=20)
plt.hlines(1,0,T*dt,colors = "r",linewidth=5, linestyles = "dashed")
plt.text(0,1.05,'$U_{thr}$',fontsize=20)
plt.legend(['$U_1$','$U_2$','$U_3$','$U_4$','$U_5$'],fontsize=20,loc='lower right')


#Figure 3e
from matplotlib import transforms
from matplotlib.gridspec import GridSpec
def np_move_avg(a,n,mode="same"):
    return(np.convolve(a, np.ones((n,))/n, mode=mode))
hidden_shape=100
P=3
model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P,'double')
_=model.load_state_dict(torch.load(save_path+'model/H_{}_P_{}_1.pth'.format(hidden_shape,P)))
_=model.to(device)
input_data,targets=generate_inputs()
with torch.no_grad():
    o=model(input_data).flatten(0)
spk=model.S.flatten(1).cpu()
fig = plt.figure(facecolor="w", figsize=(12, 8))
gs = GridSpec(3, 3, figure=fig)
ax = fig.add_subplot(gs[0:2,0:2])
splt.raster(spk, ax, s=10, c="chocolate")
plt.gca().xaxis.set_visible(False)
plt.ylabel('Neuron',fontsize=30)
plt.fill_between(np.arange(50,250),-10,110,color='grey',alpha=0.3)
plt.ylim(-5,105)
plt.xlim(-10,510)
ax = fig.add_subplot(gs[2,0:2])
s1=plt.plot(torch.arange(spk.shape[0]),np_move_avg(spk.sum(1),1)/100*100,c='tan',linewidth=2,label='firing rate \n     (HZ)')
plt.gca().set_xticklabels(['0','0','20','40','60','80','100'])
plt.xlabel('Time (ms)',fontsize=30)
plt.ylabel('Firing fraction\n(%)',fontsize=20)
plt.fill_between(np.arange(50,250),-1,100,color='grey',alpha=0.3)
plt.ylim(-1,5)
#plt.ylim(-0.5,2)
plt.xlim(-10,510)
ax = fig.add_subplot(gs[0:2,2:])
base = plt.gca().transData
rot = transforms.Affine2D().rotate_deg(-90)
s2=plt.plot(torch.arange(spk.shape[1]),spk.sum(0)/0.1,label='firing times',linewidth=2,c='wheat',transform=rot + base)
plt.gca().yaxis.set_visible(False)
plt.ylim(5,-105)
plt.xlabel('Firing rate\n(HZ)',fontsize=20)
fig.subplots_adjust(wspace=0,hspace=0)
lns = s1
labs = [l.get_label() for l in lns]