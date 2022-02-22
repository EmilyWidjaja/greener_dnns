import os
from default_variables import *
import dense

epochs = 1
train_more = False

input()
for m, n in mn:
    net = dense.Dense(m, n, lr, momentum, device_number, experiment_name, path)
    net.main(epochs, train_path, test_path, train_more)

os.system('tput bel')
print('Complete')