#Runs test_loop to ensure correct functionality

import os
from kernel_vars import *

from models import measure_kernel_energy
import models

epochs = 1
train_more = False
kers = [kers[0]]

for ker in kers:
    net = models.FashionConv(ker, out_channels, lr, momentum, device_number, path=path, experiment_name=experiment_name, fileformat=experiment_name)
    net.main(epochs, train_path, test_path, train_more)

os.system('tput bel')
print('Training complete')

# Take measurements
# from kernel_vars import *
# from models import measure_kernel_energy
import csv

#Load data
debug = True
attributes = 'timestamp,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,temperature.gpu,memory.used,pstate' #specify attributes to be recorded

mes = measure_kernel_energy(device_number, times, trials, path, sampling_period, momentum, lr, \
    experiment_name, debug, attributes)
print('Test_path ')
print(test_path)
mes.main(kers, out_channels, train_path, test_path, warm_up_times)
