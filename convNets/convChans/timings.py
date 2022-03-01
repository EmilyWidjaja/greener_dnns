from chan_vars import *
from measure_chan_energy import measure_chan_energy

#Load data
debug = True
attributes = 'timestamp,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,temperature.gpu,memory.used,pstate' #specify attributes to be recorded

mes = measure_chan_energy(device_number, times, trials, path, sampling_period, momentum, lr, \
    experiment_name, debug, attributes)

mes.main(ker, chans, train_path, test_path, warm_up_times)