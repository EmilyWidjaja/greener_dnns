#Specify variables for that experiment here
#Housekeeping
import sys
parentdir = '/home/emily/GreenerDNNs'
print("Importing channel variables...")
sys.path.append(parentdir) 

train_path = '/home/emily/GreenerDNNs/convNets/convKernels/train_loader.py'
test_path = '/home/emily/GreenerDNNs/convNets/convKernels/test_loader.py'
path = '/home/emily/GreenerDNNs/convNets/convChans'
experiment_name = 'chan'
timing_name = 'chan'
#Fileformat needs to be set for each specific class in Model.py
device_number = 1
device_name = 'A5000'

#Training variables
data_size = (1, 28, 28)
batch_size_train = 60000
batch_size_test = 10000
lr = 0.05
momentum = 0.3
ker = 3
chans = range(1, 17)


#Test variables
times = 1
trials = 5
warm_up_times = 10
sampling_period = 20

if __name__ == "__main__":
    print("Setting up...")

    #Check for data 
    if train_path.is_file():
        print('Train data found.')
    else:
        print('Train data not found.')
        raise SystemExit(0)

    if test_path.is_file():
        print('Test data found.')
    else:
        print('Test data not found.')
        raise SystemExit(0)
