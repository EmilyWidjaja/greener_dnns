from act_vars import *
from custom_resnet import measure_act_energy
import torch
from dataloader import CustomDataset

#Load data
debug = True
attributes = 'timestamp,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,temperature.gpu,memory.used,pstate' #specify attributes to be recorded

trainloader = torch.load('./trainloader.pth')
testloader = torch.load('./testloader.pth')

mes = measure_act_energy(device_number, times, trials, path, sampling_period, momentum, lr, \
    experiment_name, debug, trainloader, testloader, attributes)

mes.main(activations, train_path, test_path, timing_name, warm_up_times)