#Specify variables for that experiment here

#Housekeeping
import sys
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))
parentdir = '/home/emily/GreenerDNNs'
sys.path.append(parentdir) 

train_path = './train_loader.py'
test_path = './test_loader.py'
path = '/home/emily/GreenerDNNs/denseNets'
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
