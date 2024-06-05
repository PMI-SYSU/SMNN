import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

from SMNN.Figure.mnist_plot import *
from SMNN.mnsit.model import *
from SMNN.Figure.mnsit_utils import *
save_path='Results/mnist/' #save path for SMNN
def plot_attractor(model,target,a=30,b=-60,C=30,d=-60):    
    # 绘制散点图
    fig1 = plt.figure(figsize=(15,8),dpi=200)
    ax1 = fig1.add_subplot(121,projection='3d')
    ax2 = fig1.add_subplot(122,projection='3d')
    loader = DataLoader(mnist_test, batch_size=1, shuffle=True)
    
    cms=['Reds','Blues','Oranges']
    colors=['#d62728','#1f77b4','#ff7f0e']
    for i in range(1000):
        data,targets_onehot,targets=generate_input(loader)
        if targets[0] != target[0]:
            cm=cms[0]
            c=colors[0]
        elif targets[0] != target[1]:
            cm=cms[1] 
            c=colors[1]
        elif targets[0] != target[2]:
            cm=cms[2]
            c=colors[2]
        else:    
            continue
        with torch.no_grad():
            _=model(data)
        if 'rate' in model._get_name():
            rin=torch.matmul(model.pin.T,model.U.max(0).values).cpu().detach()
            rout=torch.matmul(model.pout.T,model.U.max(0).values).cpu().detach()
        elif 'MDL' in model._get_name():
            rin=torch.matmul(model.pin.T,model.R.max(0).values).cpu().detach()
            rout=torch.matmul(model.pout.T,model.R.max(0).values).cpu().detach()
        x,y,z=torch.split(rin,1,dim=-2)
        s1=ax1.scatter(x, y, z,c=c,alpha=0.5,s=50)
        x,y,z=torch.split(rout,1,dim=-2)
        s2=ax2.scatter(x, y, z,c=c,alpha=0.5,s=50) 
    
    for i in range(3):
        ax2.scatter([], [], [],c=colors[i],label='number {}'.format(target[i]),s=50) 

    ax1.set_title('$r_{in}$',fontsize=30)
    ax2.set_title('$r_{out}$',fontsize=30)

    ax2.legend(loc=(-0.45,0.65),fontsize=25,markerscale=1.5)

    ax1.view_init(a,b)
    ax2.view_init(C,d)

for hidden_shape in [100]:
    for P in [3]:
        model=MDL_RNN_mnist(input_shape,hidden_shape,output_shape,P,'double')
        _=model.load_state_dict(torch.load(save_path+'model/H_{}_P_{}_double.pth'.format(hidden_shape,P)))
        _=model.to(device)
        _=model.eval()
        plot_attractor(model,[8,1,9])

