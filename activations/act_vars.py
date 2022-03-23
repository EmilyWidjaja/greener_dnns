#Specify variables for that experiment here
#Housekeeping
import sys
parentdir = '/home/emily/GreenerDNNs'
print("Importing activations variables...")
sys.path.append(parentdir) 

#Technically unnecessary - train & test loaders created in dataloader.py
train_path = '/home/emily/GreenerDNNs/activations/train_loader.py'
test_path = '/home/emily/GreenerDNNs/activations/test_loader.py'

path = '/home/emily/GreenerDNNs/activations/'
experiment_name = 'activations'
timing_name = 'activations'
#Fileformat needs to be set for each specific class in Model.py
device_number = 1
device_name = 'A5000'

#Training variables
data_size = (3, 256, 256)
batch_size_train = 3925-785
batch_size_test = 785
lr = 0.05
momentum = 0.3


#Test variables
times = 1
trials = 5
warm_up_times = 10
sampling_period = 20
activations = ['none', 'relu', 'leaky', 'tanh', 'elu']

