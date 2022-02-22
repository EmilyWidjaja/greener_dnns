import torch.nn as nn
import torch.nn.functional as F
from fashion_base import FashionVal

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
    def __init__(self, no_layers, no_nodes, lr, momentum, device_number, experiment_name, path='/home/emily/fashion/fashion_module/'):
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

        
