import sys
parentdir = '/home/emily/GreenerDNNs'
print("Importing DVFS variables...")
sys.path.append(parentdir) 

from torchvision import models
from torch import nn
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from DVFS_helpers import *
from sklearn.model_selection import train_test_split
from torch import optim
import torch.nn.functional as F

#Define device
device_number = 1
torch.cuda.set_device(device_number)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(device), ' device used.')
if device == 'cpu':
    raise SystemExit(0)
resnet18 = models.resnet18(pretrained=True)

##Freeze layers
for param in resnet18.parameters():
    param.requires_grad = False

#Modify last layer
number_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(number_features, 10)
resnet18.to(device)
print('Resnet18 loaded & modified')

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.1, momentum=0.3)

#Define device



#Load images
"""Data Loading and normalization"""
image_path = '/home/emily/resnet/Resnet/val'
os.chdir(image_path)
folders = os.listdir(image_path)
images = []
labels = []
data_dict = {}
encoder = {'chainsaw': 0, 'golf': 1, 'tench':2, 'french_horn': 3, 'springer': 4, 'church': 5, \
    'cassette': 6, 'truck': 7, 'parachute': 8, 'pump': 9}
decoder = {0: 'chainsaw', 1: 'golf', 2: 'tench', 3: 'french_horn', 4: 'springer', 5: 'church', \
    6: 'cassette', 7: 'truck', 8: 'parachute', 9: 'pump'}

count = 0
print('Normalizing images...')
for folder in tqdm(folders):
    if "." in folder:
        continue
    cwd = os.path.join(image_path, folder)
    os.chdir(cwd)
    files = [f for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))]
    data_dict[folder] = 0
    for file in files:
        if ".DS_Store" in file:
            continue
        image = Image.open(file)
        count += 1
        image = normalize(image)
        images.append(image)
        labels.append(encoder[folder])
        data_dict[folder] += 1

print(data_dict)
print("Data retrieval and standardization complete.")
# Change B&W images
print(type(images[0]))
print(images[0].size())

bw = []
idx = 0
while idx < len(images):
    #Find indexes of B&W images and remove
    if images[idx].shape[0] == 1:  #If first dimension has 1 channel - i.e. B&W
        new_img = np.vstack((images[idx],)*3)
        images[idx] = torch.FloatTensor(new_img)
        continue
    idx += 1

idxs = []
wrong = 0
for idx, img in enumerate(images):
    if type(image) != torch.Tensor:
        print('not a tensor!', type(image))
        idxs.append(idx)
        wrong += 1

print('wrong: ', wrong)
print(type(images))
images = torch.stack(images)
#Create dataloader
print("Creating dataloader...")
print('Images size: ', len(images), type(images[0]), images[0].size())
print('Labels size: ', len(labels), type(labels[0]))
if len(images) != len(labels):
    print('ERROR: # of images != # of labels. Terminating program...')
    raise SystemExit(0)

# input('continue?')
test_split = 0.2
test_no = round(test_split*len(images))
train_no = len(images) - test_no

dataset = CustomDataset(images, labels)
train_data, test_data = torch.utils.data.random_split(dataset, [test_no, train_no], generator=torch.Generator().manual_seed(42))


#Split into train and test
trainloader = torch.utils.data.DataLoader(train_data, batch_size=train_no, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=test_no, shuffle=True)
print("Dataloader created.\n")

#Test train & test
#Train resnet18 
def test():
    resnet18.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data = data.to(device)
            target = target.to(device)
            output = resnet18(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss /= len(testloader.dataset)
    accuracy = 100.*correct/len(testloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{}\
            ({corr:.0f}%)\n'.format(test_loss, correct, len(testloader.dataset), corr=100.*correct/len(testloader.dataset)))
    return accuracy, test_loss

def train_model(num_epochs=25):
    for epoch in range(num_epochs):
       print('Epoch {}/{}'.format(epoch, num_epochs - 1))
       print('-' * 10)

       #set model to trainable
       resnet18.train()
       
       for i, (inputs, labels) in enumerate(trainloader):
           inputs = inputs.to(device)
           labels = labels.type(torch.LongTensor)
           labels = labels.to(device)

           optimizer.zero_grad()
           output = resnet18(inputs)

           loss = F.cross_entropy(output, labels)

           torch.cuda.empty_cache()
           loss.backward()
           optimizer.step()
           print(f'Train epoch: {epoch}\t loss: {loss:.6f}')
           print('Nothing')
       # Iterate over data.
       test()
    return



"""START TRAINING"""
num_epochs = 40
train_model(num_epochs)
test()

torch.save(resnet18.state_dict(), '/home/emily/GreenerDNNs/DVFS/resnet18/resnet18.pth')
torch.save(optimizer.state_dict(), '/home/emily/GreenerDNNs/DVFS/resnet18/res_optimizer.pth')
print('Model saved')