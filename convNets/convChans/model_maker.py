import os
from chan_vars import *
import convNets.convKernels.models as models
from tqdm import tqdm

epochs = 20
train_more = False

for chan in tqdm(chans):
    #Check it can access the right models (being convKernels)
    net = models.FashionConv(ker, chan, lr, momentum, device_number, path=path, experiment_name=experiment_name, fileformat=experiment_name+str(chan))
    net.main(epochs, train_path, test_path, train_more)

os.system('tput bel')
print('Training complete')