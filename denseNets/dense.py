import torch.nn as nn
import torch.nn.functional as F
from timing_class import measure_energy
from fashion_base import FashionVal
from tqdm import tqdm
import os

"""
Classes to override:
class Net(nn.Module)

class Net
class FashionVal():
    def __init__(self, no_layers, no_nodes, lr, momentum, path):
    def instantiate_net(self, vars):

        """
#Set Network
class Net(nn.Module):
    def __init__(self, no_layers, no_nodes):
        #instantiates DNN with no_layers fully connected layers, a flattened input layer and a 10-node output layer
        super(Net, self).__init__()
        n = no_nodes
        
        #Define topology
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features=28*28, out_features=n))
        self.d0 = nn.Linear(in_features=28*28, out_features=n)
        for i in range(0, no_layers):
            self.layers.append(nn.Linear(in_features=n, out_features=n))
        self.layers.append(nn.Linear(in_features=n, out_features=10))
        print("Model Architecture: {} hidden layers of {} nodes.".format(no_layers, n))

    def forward(self, x): 
        x = x.reshape(-1, 28*28).cuda()
        for layer in self.layers:
            x = layer(x)
        return F.log_softmax(x)

class Dense(FashionVal):
    def __init__(self, no_layers, no_nodes, lr, momentum, device_number, experiment_name, path):
        super(Dense, self).__init__(lr, momentum, device_number, experiment_name, path)
        #set constants
        self.fileformat = '{}x{}'.format(no_layers, no_nodes)
        self.no_layers = no_layers
        self.no_nodes = no_nodes
        
        return

    def instantiate_net(self):
        #Explicitly overwrite instantiate net. Needs to set self.network with right net
        self.network = Net(self.no_layers, self.no_nodes)
        return

        
class measure_dense_energy(measure_energy):
    def __init__(self, device_number, times, trials, path, sampling_period, momentum, lr, exp, debug, attributes=''):
        super(measure_dense_energy, self).__init__(device_number, times, trials, path, sampling_period, momentum, lr, exp, debug, attributes)
        return

    def load_model(self, m=0, n=0, default=False):
        net_obj = Dense(m, n, self.lr, self.momentum, self.device_number, self.exp, self.path)
        if default:
            self.default_net = net_obj
        return net_obj

    def main(self, MN, train_path, test_path, timing_name, warm_up_times=0):
        #Warm-up
        self.set_device()
        net_obj = self.load_model(106, 106, default=True) #Just reminder to set a default!
        self.load_infer_data(train_path, test_path)
        self.instantiate_model(net_obj)
        self.warm_up(net_obj, warm_up_times)

        #Take readings
        for m, n in tqdm(MN):
            for trial in range(0, self.trials):
                net_obj = self.load_model(m, n)
                self.instantiate_model(net_obj)
                command_str = self.define_command(trial, net_obj, timing_name)
                self.test(command_str, net_obj)


        print('Timings complete.')

        if self.debug:
            self.sampling_iterator(timing_name)
            
        os.system('tput bel')
        return

