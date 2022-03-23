from act_vars import *
from custom_resnet import measure_act_energy

#Load data
debug = True
attributes = 'timestamp,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,temperature.gpu,memory.used,pstate' #specify attributes to be recorded

mes = measure_act_energy(device_number, times, trials, path, sampling_period, momentum, lr, \
    experiment_name, debug, attributes)

mes.main(activations, train_path, test_path, timing_name, warm_up_times)