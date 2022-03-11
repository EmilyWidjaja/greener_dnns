#Specify variables for that experiment here

#Housekeeping
import sys
import os
parentdir = '/home/emily/GreenerDNNs'
print("Importing DVFS variables...")
sys.path.append(parentdir) 
from DVFS_helpers import get_supported_clocks

clock_path = '/home/emily/resnet/clock_freqs/'
path = '/home/emily/GreenerDNNs/DVFS'
image_path = '/home/emily/resnet/Resnet/val'
old_path = '/home/emily/resnet/Resnet/'
experiment_name = 'dvfs'
device_number = 1
device_name = 'A5000'

#Training variables
data_size = (3, 50, 50)
batch_size_train = 0
batch_size_test = 15000
MEM_CLOCKS = [7601, 5001, 810] #(8001 and 450 don't work or don't have much range respectively)
GR_CLOCKS = get_supported_clocks(os.path.join(clock_path, 'supported_gr_clocks.csv'), header=False)

print("Clocks loaded. Mem_CLOCKS: ", MEM_CLOCKS[0:10], "...")
print("GR_CLOCKS: ", GR_CLOCKS[0:10], "...")

clocks = []
for mem in MEM_CLOCKS:
    for gr in GR_CLOCKS:
        clocks.append((mem, gr))

#Test variables
times = 50
warm_up_times = 5
trials = 5
exp = 'dvfs1'
debug = False
sampling_period = 30
attributes = []