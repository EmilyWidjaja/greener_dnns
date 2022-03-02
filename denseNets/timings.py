from default_variables import *
from dense import measure_dense_energy
import csv



#Load data
debug = False
attributes = 'timestamp,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,temperature.gpu,memory.used,pstate' #specify attributes to be recorded

with open('repeats.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
mn = []
for m, n in data:
    mn.append((int(m), int(n)))

print(mn)
print(len(mn))

mes = measure_dense_energy(device_number, times, trials, path, sampling_period, momentum, lr, \
    experiment_name, debug, attributes)
mes.main(mn, train_path, test_path, timing_name, warm_up_times)