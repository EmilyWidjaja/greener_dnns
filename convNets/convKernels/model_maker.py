import os
from kernel_vars import *
import models
from tqdm import tqdm

epochs = 60
train_more = False

def already_trained(fileformat):
    files = os.listdir(os.path.join(path, 'train_data', experiment_name))
    if fileformat in files:
        return True
    else:
        return False

for ker in tqdm(kers):
    print('Loading kernel {}'.format(ker))
    if already_trained(experiment_name+str(ker)+'.csv'):
        print('Already trained. Skipping...\n')
        continue
    print('Begin training...')
    net = models.FashionConv(ker, out_channels, lr, momentum, device_number, \
        pad1=pad1, path=path, experiment_name=experiment_name, fileformat=experiment_name+str(ker))
    net.batch_size_train = batch_size_train; net.data_size = data_size
    net.main(epochs, train_path, test_path, train_more)

os.system('tput bel')
print('Training complete')