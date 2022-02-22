#Specify variables for that experiment here - THIS IS AN EXAMPLE

#Housekeeping
import sys
parentdir = '/home/emily/GreenerDNNs'
sys.path.append(parentdir)
print('you are loading thr wrong variables')

train_path = './train_loader.py'
test_path = './test_loader.py'
path = '/home/emily/fashion/'
experiment_name = 'dense'
#Fileformat needs to be set for each specific class in Model
device_number = 1
device_name = 'A5000'

#Training variables
data_size = (1, 28, 28)
batch_size_train = 60000
batch_size_test = 10000
lr = 0.1
momentum = 0.3
layers = range(1, 151, 5)
mn = []
for l in layers:
    mn.extend([(l, i) for i in range(1, 151, 5)])

#Test variables
times = 50
warm_up_times = 10
sampling_period = 15
