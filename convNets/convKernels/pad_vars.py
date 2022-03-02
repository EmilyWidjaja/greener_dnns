#Specify variables for that experiment here
import os

#Housekeeping
import sys
parentdir = '/home/emily/GreenerDNNs'
print("Importing kernel variables...")
sys.path.append(parentdir) 

train_path = '/home/emily/GreenerDNNs/convNets/convKernels/train_loader.py'
test_path = '/home/emily/GreenerDNNs/convNets/convKernels/test_loader.py'
path = '/home/emily/GreenerDNNs/convNets/convKernels'
experiment_name = 'pad'
timing_name = 'pad'
#Fileformat needs to be set for each specific class in Model.py
device_number = 1
device_name = 'A5000'

#Training variables
data_size = (1, 28, 28)
batch_size_train = 60000
batch_size_test = 10000
lr = 0.1
momentum = 0.3
out_channels = 16
kers = range(1, 15, 2)
pad1 = 'same'


#Test variables
times = 7
trials = 5
warm_up_times = 10
sampling_period = 20

if __name__ == "__main__":
    print("Setting up...")

    #Check for data 
    print('Train data found: {}', os.path.join(path, 'train_loader.py').is_file())
    print('Test data found: {}', os.path.join(path, 'test_loader.py').is_file())
