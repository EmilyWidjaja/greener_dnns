#Specify variables for that experiment here

#Housekeeping
import sys
import os
parentdir = '/home/emily/GreenerDNNs'
print("Importing DVFS variables...")
sys.path.append(parentdir) 
from DVFS_helpers import get_supported_clocks

path = '/home/emily/GreenerDNNs/DVFS'
clock_path = os.path.join(path, 'current_clocks')
image_path = '/home/emily/GreenerDNNs/denseNets/val'
experiment_name = 'dvfs'
device_number = 0
device_name = 'TitanX'

#Training variables
data_size = (3, 50, 50)
batch_size_train = 0
batch_size_test = 15000
if os.path.isfile(os.path.join(clock_path, 'supported_gr_clocks.csv')):
    GR_CLOCKS = get_supported_clocks(os.path.join(clock_path, 'supported_gr_clocks.csv'), header=False)
    MEM_CLOCKS = get_supported_clocks(os.path.join(clock_path, 'supported_mem_clocks.csv'), header=False)
    MEM_CLOCKS = [7601, 5001, 810]
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
exp = 'dvfs'
debug = True
sampling_period = 30
attributes = []

#get memory and core clocks
if __name__ == "__main__":
    os.system('nvidia-smi --query-supported-clocks=mem -i {} --format=csv,noheader,nounits > supported_mem_clocks.csv'.format(device_number))
    os.system('nvidia-smi --query-supported-clocks=gr -i {} --format=csv,noheader,nounits > supported_gr_clocks.csv'.format(device_number))
    os.system('mkdir current_clocks')
    os.system('nvcc -o change_clocks nvml_run.cu -I/usr/local/cuda-11.6/targets/x86_64-linux/include -L/usr/local/cuda/lib64 -lnvidia-ml')
    os.system('nvcc -o reset_clocks reset_clocks.cu -I/usr/local/cuda-11.6/targets/x86_64-linux/include -L/usr/local/cuda/lib64 -lnvidia-ml')