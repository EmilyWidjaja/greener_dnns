import os

import torch
from act_vars import *
from dataloader import CustomDataset
from custom_resnet import Activation
from tqdm import tqdm
import pandas as pd

trainloader = torch.load('./trainloader.pth')
testloader = torch.load('./testloader.pth')
epochs = 20
train_more = False
dict = {}

for act in tqdm(activations):
    #Check it can access the right models (being convKernels)
    net = Activation(lr, momentum, act, device_number, experiment_name, path, trainloader, testloader)
    net.set_batch_size_train(batch_size_train, data_size)
    best_acc, best_test_loss = net.main(epochs, train_path, test_path, train_more)
    dict[act] = [best_acc, best_test_loss]

df = pd.DataFrame(dict)
df.columns = ['activations', 'accuracy', 'test loss']

print(df)
os.system('tput bel')
print('Training complete')