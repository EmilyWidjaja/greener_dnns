import torch.nn as nn
import torch.nn.functional as F
from timing_class import measure_energy
from fashion_base import FashionVal
import torch
import os
from tqdm import tqdm

class ConvNet(nn.Module):
    def __init__(self, kernel_size, out_channels1, strides, pad1, dilation, input_dims):
        super(ConvNet, self).__init__()
        #Find padding size for same:
        if pad1 == 'same':
            ker = kernel_size[0]
            if ker % 2 != 0:
                pad1 = (ker//2, ker//2, ker//2, ker//2)
            else:
                pad1 = (ker // 2, ker - (ker // 2) - 1, ker // 2, ker - (ker // 2) - 1) #Left, Right, Top, Bottom (0, 0, 0, 0)
        else:
            pad1 = (0, 0, 0, 0)

        #Find output size before flattening
        # print("Kernel_size = {}".format(kernel_size))
        alist = [input_dims[1], pad1[0], dilation[0], kernel_size[0], strides[0]]
        for idx, el in enumerate(alist):
            if type(el) != int:
                print('wrong', type(el), idx)
                print(strides)
        conv1_h=(input_dims[1]+2*pad1[0]-dilation[0]*(kernel_size[0]-1)-1)/strides[0]+1
        conv1_w = (input_dims[2]+2*pad1[2]-dilation[1]*(kernel_size[1]-1)-1)/strides[1]+1
        conv1_dims=(out_channels1, conv1_h, conv1_w)
        # print('conv1 size: {}'.format(conv1_dims))

        ker_p = (2,2)
        pool_h = conv1_h / ker_p[0]
        pool_w = conv1_w / ker_p[1]
        self.flat_dims = int(pool_h * pool_w * out_channels1)
        
        # print('pool size: ({}, {}, {})'.format(out_channels1, pool_h, pool_w))
        # self.flat_dims = out_channels1*13*13

        #Define model architecture
        #Opt padding
        self.pad = nn.ZeroPad2d(pad1)    #Left, Right, Top, Bottom (0, 0, 0, 0)

        #Conv layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels1, kernel_size=kernel_size, stride=strides) # Padding implemented manually in padding layer
        #Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=ker_p, stride=None)

        #2 Dense layers
        self.d1 = nn.Linear(in_features=self.flat_dims, out_features=100)
        self.d2 = nn.Linear(in_features=100, out_features=10)
        return
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        torch.cuda.empty_cache()
        x = F.relu(x)
        torch.cuda.empty_cache()
        # print('Size of conv data = {}'.format(x.size()))
        x = self.pool(x)

        # print('Size of pool data = {}'.format(x.size()))
        x = x.reshape(-1, self.flat_dims).cuda()
        # print('Type of data: {}, {}'.format(type(x), type(x[0][0].item())))
        # print('Size of data = {}'.format(x.size()))
        
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        return F.log_softmax(x)


class FashionConv(FashionVal):
    def __init__(self, kernel_size=3, out_channels1=16, lr=0.01, momentum=0.9, device_number=1, strides=(1,1), pad1=0, experiment_name='', path='/home/emily/fashion/fashion_module/', fileformat='chan'):
        super(FashionConv, self).__init__(lr, momentum, device_number, experiment_name, path=path)
        if type(kernel_size) == tuple:
            self.kernel_size = kernel_size
        else:
            self.kernel_size = (kernel_size, kernel_size)
        self.strides = strides
        self.out_channels1 = out_channels1

        #Find padding size
        if pad1 == 'same':
            self.pad = pad1
        else:
            self.pad = (pad1, pad1, pad1, pad1)

        self.fileformat = fileformat + '{}'.format(kernel_size)
        print(self.fileformat)
        return

    def instantiate_net(self):
        self.network = ConvNet(kernel_size=self.kernel_size, out_channels1=self.out_channels1, \
            strides=self.strides, pad1=self.pad, dilation=(1,1), input_dims=(1,28,28))
        return 

class measure_kernel_energy(measure_energy):
    def __init__(self, device_number, times, trials, path, sampling_period, momentum, lr, exp, debug, attributes=''):
        super(measure_kernel_energy, self).__init__(device_number, times, trials, path, sampling_period, momentum, lr, exp, debug, attributes)
        return

    def load_model(self, ker, out_channels1, default=False):
        net_obj = FashionConv(ker, out_channels1, self.lr, self.momentum, self.device_number, experiment_name=self.exp, path=self.path, fileformat=self.exp)
        if default:
            self.default_net = net_obj
        return net_obj

    def main(self, kers, out_channels1, train_path, test_path, warm_up_times=0):
        #Warm-up
        self.set_device()
        net_obj = self.load_model(3, out_channels1, default=True) #Just reminder to set a default!
        self.load_infer_data(train_path, test_path)
        self.instantiate_model(net_obj)
        self.warm_up(net_obj, warm_up_times)

        #Take readings
        if self.debug and len(kers) >= 4:
            kers = kers[-2::]# + kers[-2::]
        for ker in tqdm(kers):
            for trial in range(0, self.trials):
                net_obj = self.load_model(ker, out_channels1)
                self.instantiate_model(net_obj)
                command_str = self.define_command(trial, net_obj)
                self.test(command_str, net_obj)

        print('Timings complete.')
        os.system('tput bel')
        return