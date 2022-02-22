#Loads MNIST fashion dataset into dataloaders and saves them
import torch
import torchvision
from default_variables import *

#%%Stats
global_mean = 0.5
global_sd = 0.5

#%%Load data
if __name__ == '__main__':
    train_data = torchvision.datasets.FashionMNIST('./data', train=True, download=True, \
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((global_mean,), (global_sd,))
            ]
        ))
    test_data = torchvision.datasets.FashionMNIST('./data', train=False, download=True, \
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((global_mean,), (global_sd,))
            ]
        )) 

    #%%
    #Batch Data
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=60000, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=True)

    #-----------
    # print(data, target)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=True)

    torch.save(train_loader, './train_loader.py')
    torch.save(test_loader, './test_loader.py')
    print('Dataloaders created: \nTrain_loader: {} batch(es) of {}\nTest_loader: {} batch(es) of {}'.format(len(train_loader), \
        batch_size_train, len(test_loader), batch_size_test))


#------------------
#Load examples
# examples=enumerate(test_loader)     #load some examples
# batch_idx, (example_data, example_targets) = next(examples)

# print("shape: ", example_data.shape)

# import matplotlib.pyplot as plt

# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title(f"Ground Truth: {example_targets[i]}")
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
# %%
