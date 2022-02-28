import torch.nn as nn
import torch.nn.functional as F
from timing_class import measure_energy
from fashion_base import FashionVal
import torch

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
        conv1_h=(input_dims[1]+2*pad1[0]-dilation[0]*(kernel_size[0]-1)-1)/strides[0]+1
        conv1_w = (input_dims[2]+2*pad1[2]-dilation[1]*(kernel_size[1]-1)-1)/strides[1]+1
        conv1_dims=(out_channels1, conv1_h, conv1_w)
        # print('conv1 size: {}'.format(conv1_dims))

        # pad_p = (0,0)
        # dil_p = (1,1)
        ker_p = (2,2)
        # stride_p = (1,1)
        # pool_h = ((conv1_h + 2*pad_p[0]-dil_p[0]*(ker_p[0]-1)-1)/stride_p[0])+1
        # pool_w = ((conv1_w + 2*pad_p[1]-dil_p[1]*(ker_p[1]-1)-1)/stride_p[1])+1
        pool_h = conv1_h / ker_p[0]
        pool_w = conv1_w / ker_p[1]
        self.flat_dims = pool_h * pool_w * out_channels1
        
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
        print('Type of data: {}, {}'.format(type(x), type(x[0][0].item())))
        print('Size of data = {}'.format(x.size()))
        
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        return F.log_softmax(x)


class FashionConv(FashionVal):
    def __init__(self, kernel_size=3, out_channels1=16, lr=0.01, momentum=0.9, device_number=1, strides=(1,1), pad1=0, path='/home/emily/fashion/fashion_module/', fileformat='chan'):
        super(FashionConv, self).__init__(lr, momentum, device_number, path=path)
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