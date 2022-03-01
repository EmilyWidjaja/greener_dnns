import os
from kernel_vars import *
import models
from tqdm import tqdm

epochs = 20
train_more = False

for ker in tqdm(kers):
    net = models.FashionConv(ker, out_channels, lr, momentum, device_number, path=path, experiment_name=experiment_name, fileformat=experiment_name+str(ker))
    net.main(epochs, train_path, test_path, train_more)

os.system('tput bel')
print('Training complete')