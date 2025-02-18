import torch 
from snntorch import surrogate

"""

Model settings for contextual-dependent task

P: mode size
hidden_shape: number of neurons
input_shape: numbers of input signals
output_shape: output size
n: trials of training
epochs_num: epochs for each trials
zero_time1: zero time beform stimulus
zero_time2: zero time after stimulus
target_zero: target output of zero
T: time steps
zero1: zero input beform stimulus
zero2: zero input beform stimulus
batchsize: batch size for training
device: cpu or gpu device for running
spike_grad: surrogate delta function
spike_grad_approx : smooth transfer function
optimizer: optimizer
vthr : firing threshold
tau_m : membrane time constant
tau_d : synaptic decay time constant
tau_r : synaptic rise time constant
dt : time step

"""

spike_grad = surrogate.fast_sigmoid()
def spike_grad_approx(x,beta=20):
    return (1+torch.tanh(beta*x))/2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
input_shape=4
output_shape=1
batchsize=300
n=10
epochs_num = 5
T=500
P=3
dt=0.2
hidden_shape=100
zero_time1=int(T/10)
zero_time2=int(T/2)
target_zero=torch.zeros(size=(1,1)).repeat(T-zero_time2,1,1).to(device).to(torch.float)
zeros1=torch.zeros(size=(zero_time1,input_shape,1)).to(device)
zeros2=torch.zeros(size=(zero_time2,input_shape,1)).to(device)
optimizer = torch.optim.Adam
vthr=torch.tensor(1.0)
tau_m=torch.tensor(20)
tau_d=torch.tensor(30)
tau_r=torch.tensor(2)
