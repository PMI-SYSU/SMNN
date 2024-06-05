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

from Figure.mnist_utils import *
from SMNN.mnist.model import *

P_range=[1,2,3,4,5,10,30,50,100,200]
N_range=[30,50,100,200]
save_path='Results/mnist/' #save path for SMNN
rate_path='Results/mnist/rate/' #save path for MDL RNN
full_path='Results/mnist/full/'#save path for SNN
full_rate_path='Results/mnist/full_rate/'#save path for RNN
tref0_path='Results/mnist/tref0/'#save path for SMNN with tref=0
replace_path='Results/mnist/replace/'#save path for SMNN with approximate step function


# Figure 2a
P_range=[1,2,3,5,30,100]
N_range=[100,200]
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot()
name='acc'

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
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='SMNN $N$={}'.format(N_range[j]))
    j+=1

j=0
for hidden_shape in N_range:
    results=[]
    for P in P_range:
        acc=[]
        for i in range(3):
            a = torch.load(full_path+'{}/H_{}_P_{}_{}.pth'.format(name,hidden_shape,P,i))
            acc.append(a[-1]*100)
        results.append(acc)
    results=np.array(results)
    mean=np.mean(results)
    std=np.std(results)
    yerr = np.zeros([2,len(P_range)])
    yerr[0,:] = std
    yerr[1,:] = std
    ax.errorbar(P_range,mean.repeat(len(P_range)),yerr=yerr[:,:],ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='SNN $N$={}'.format(N_range[j]))
    j+=1

plt.legend(loc='best',ncol=2,fontsize=15,columnspacing=0.4)
plt.xscale('log')
plt.ylabel('Test acc.(%)',fontsize=30)
plt.xlabel('Mode Size ($P$)',fontsize=30)
plt.ylim(top=98)

#Figure 2b
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot()
P_range=[1,2,3,10,50,100]
name='train_loss'
for P in P_range:
    results=[]
    for N in N_range[2:3]:
        loss=[]
        for i in range(3):
            a=torch.load(save_path+'{}/H_{}_P_{}_{}.pth'.format(name,N,P,i))
            loss.append(a[:500:20])
        loss=torch.stack(loss,0).float()
        results.append(loss)
    results=torch.tensor(results[0])
    err = np.zeros([2,results.shape[1]])
    mean=results[0]
    std=results[1]

    _=ax.errorbar(np.arange(mean.shape[0])*20,mean,yerr=err[:,:],ecolor='k',elinewidth=0.5,marker='.',\
    mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='$P$={}'.format(P))
plt.legend(fontsize=25,ncol=2,columnspacing=0.4)
plt.ylabel('Cross Entropy',fontsize=30)
plt.xlabel('Training batch',fontsize=30)

#Figure 2c
P_range=[1,2,3,5,30,100]
N_range=[200]
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot()
name='acc'
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
ax=plt.gca()
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
            a = torch.load(rate_path+'{}/H_{}_P_{}_{}.pth'.format(name,hidden_shape,P,i))
            acc.append(a[-1]*100)
        results.append(acc)
    results=np.array(results)
    mean=np.array([np.mean(x) for x in results])
    std=np.array([np.std(x) for x in results])
    yerr = np.zeros([2,len(P_range)])
    yerr[0,:] = std
    yerr[1,:] = std
    ax.errorbar(P_range,mean,yerr=yerr[:,:],ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='MDL RNN')
    j+=1

j=0
for hidden_shape in N_range:
    results=[]
    for P in P_range:
        acc=[]
        for i in range(3):
            a = torch.load(full_path+'{}/H_{}_P_{}_{}.pth'.format(name,hidden_shape,P,i))
            acc.append(a[-1]*100)
        results.append(acc)
    results=np.array(results)
    mean=np.mean(results)
    std=np.std(results)
    yerr = np.zeros([2,len(P_range)])
    yerr[0,:] = std
    yerr[1,:] = std
    ax.errorbar(P_range,mean.repeat(len(P_range)),yerr=yerr[:,:],ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='SNN')
    j+=1

j=0
for hidden_shape in N_range:
    results=[]
    for P in P_range:
        acc=[]
        for i in range(3):
            a = torch.load(full_rate_path+'{}/H_{}_P_{}_{}.pth'.format(name,hidden_shape,P,i))
            acc.append(a[-1]*100)
        results.append(acc)
    results=np.array(results)
    mean=np.mean(results)
    std=np.std(results)
    yerr = np.zeros([2,len(P_range)])
    yerr[0,:] = std
    yerr[1,:] = std
    ax.errorbar(P_range,mean.repeat(len(P_range)),yerr=yerr[:,:],ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='RNN')
    j+=1


plt.ylim(94.5,99)
plt.legend(loc='best',ncol=2,fontsize=15,columnspacing=0.4)
plt.xscale('log')
plt.ylabel('Test acc.(%)',fontsize=30)
plt.xlabel('Mode Size ($P$)',fontsize=30)

#Figure 2d
model=MDL_RNN_mnist(input_shape,100,output_shape,10)
_=model.load_state_dict(torch.load(save_path+'model/H_{}_P_{}_1.pth'.format(100,10)))
_=model.to(device)
loader = DataLoader(mnist_test, batch_size=1, shuffle=True)
data,_,_=generate_input(loader)
with torch.no_grad():
    o=model(data)
T=100
U=model.U.cpu().detach()
r=model.R.cpu().detach()
r=r.reshape(r.shape[0],-1)
U=U.reshape(U.shape[0],-1)
index=torch.argmax(r,dim=0)
index=index[U.min(dim=0).values>=-1]
U=U[:,U.min(dim=0).values>=-1]
Um=U[index,:].diag()

a=0
b=3
U=U[:,a:b]
dt=0.2
U=torch.where(U>1,1,U)
Um=torch.where(Um>1,1,Um)

fig = plt.figure(facecolor="w", figsize=(12,8),dpi=200)
ax = fig.add_subplot(111)
plt.plot(np.arange(T)*dt,U)
plt.xlabel('Time (ms)',fontsize=30)
plt.ylabel('U',fontsize=30)
marker=['s','v','^']
for i in range(b-a):
    plt.scatter(index[a:b][i]*dt,Um[a:b][i],marker=marker[i],s=200,alpha=1)
plt.ylim(torch.min(U)-0.1,torch.max(U)+0.1)
plt.hlines(0,0,T*dt,colors = "k",linewidth=5, linestyles = "dashed")
plt.text(0,-0.15,'$U_{res}$',fontsize=20)
plt.hlines(1,0,T*dt,colors = "r",linewidth=5, linestyles = "dashed")
plt.text(0,1.1,'$U_{thr}$',fontsize=20)
plt.ylim(top=1.6)
plt.ylim(bottom=-0.5)
plt.legend(['$U_1$','$U_2$','$U_3$','argmax($r_1$)','argmax($r_2$)','argmax($r_3$)'],fontsize=20,ncol=2)


#Figure 2e
from matplotlib import transforms
from matplotlib.gridspec import GridSpec
model=MDL_RNN_mnist(input_shape,100,output_shape,10)
_=model.load_state_dict(torch.load(save_path+'model/H_{}_P_{}_0.pth'.format(100,10)))
_=model.to(device)
loader = DataLoader(mnist_test, batch_size=1, shuffle=True)
input_data,targets,_=generate_input(loader)
with torch.no_grad():
    o=model(input_data).flatten(0)
spk=model.S.flatten(1).cpu()
fig = plt.figure(facecolor="w", figsize=(12, 8))
gs = GridSpec(3, 3, figure=fig)
ax = fig.add_subplot(gs[0:2,0:2])
splt.raster(spk, ax, s=10, c="chocolate")
plt.gca().xaxis.set_visible(False)
plt.ylabel('Neuron',fontsize=30)
plt.ylim(-5,105)
plt.xlim(-1,100)
ax = fig.add_subplot(gs[2,0:2])
s1=plt.plot(torch.arange(spk.shape[0]),spk.sum(1)/100*100,c='tan',linewidth=2,label='firing rate \n     (HZ)')
plt.gca().set_xticklabels(['0','0','4','8','12','16','20'])
plt.xlabel('Time (ms)',fontsize=30)
plt.ylabel('Firing fraction\n(%)',fontsize=20)
plt.ylim(-1,27)
plt.xlim(-1,100)
ax = fig.add_subplot(gs[0:2,2:])
base = plt.gca().transData
rot = transforms.Affine2D().rotate_deg(-90)
s2=plt.plot(torch.arange(spk.shape[1]),spk.sum(0)/0.02,label='firing times',linewidth=2,c='wheat',transform=rot + base)
plt.gca().yaxis.set_visible(False)
#plt.ylim(5,105)
plt.xlabel('Firing rate\n(HZ)',fontsize=20)
#set the xlabel and xtick on the top
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
fig.subplots_adjust(wspace=0,hspace=0)
lns = s1
labs = [l.get_label() for l in lns]
#plt.legend(lns, labs, ncol=1,loc=(0.02,-0.4),fontsize=25,frameon=True)
input_data=input_data.cpu().numpy().reshape(28,28)
plt.figure(figsize=(5,5),dpi=200)
plt.imshow(input_data,cmap='gray')
plt.gca().axis('off')
plt.title('Input image',fontsize=20)
plt.show()