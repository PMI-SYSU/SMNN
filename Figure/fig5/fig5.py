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

from SMNN.Figure.mante_plot import *
from SMNN.mante.model import *
from SMNN.Figure.mante_utils import *
save_path='Results/mante/' #save path for SMNN
#Figure 5a
for hidden_shape in [100]:
    for P in [3]:
        model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P,filter='double')
        _=model.load_state_dict(torch.load(save_path+'model/H_{}_P_{}_double.pth'.format(hidden_shape,P)))
        _=model.to(device)
        _=model.eval()
plot_attractor(model,45,100,45,100)

#Figure 5b
input_data, targets = generate_context_switch()
plot_data(input_data, targets)


#Figure 5c
for hidden_shape in [100]:
    for P in [3]:
        model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P,filter='double')
        _=model.load_state_dict(torch.load(save_path+'model/H_{}_P_{}_switch.pth'.format(hidden_shape,P)))
        _=model.to(device)
        _=model.eval()
plot_swtich_trace(model,'+1-2',45,100,45,100)

#Figure 5d
for hidden_shape in [100]:
    for P in [3]:
        model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P,'double')
        _=model.load_state_dict(torch.load(save_path+'model/H_{}_P_{}_switch.pth'.format(hidden_shape,P)))
        _=model.to(device)
        _=model.eval()
fig1 = plt.figure(figsize=(15,8),dpi=200)
ax1 = fig1.add_subplot(121,projection='3d')
input_data,targets=plot_projection(model,ax1,'+1-2',30,70)