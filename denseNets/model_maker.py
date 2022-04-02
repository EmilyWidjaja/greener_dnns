import os
from default_variables import *
import dense

epochs = 1
train_more = False

for m, n in mn:
    net = dense.Dense(m, n, lr, momentum, device_number, experiment_name, path)
    net.data_size = data_size; net.batch_train_size = batch_size_train #Set separately since feature introduced later
    net.main(epochs, train_path, test_path, train_more)

os.system('tput bel')
print('Complete')

#'/home/emily/GreenerDNNs/denseNets/results/dense/model1x1.pth'