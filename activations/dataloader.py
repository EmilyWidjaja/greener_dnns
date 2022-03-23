import os
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

"""Copy of code from DVFS since some problem loading it"""
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels):
        self.labels = labels
        self.imgs = imgs
        self.persistent_workers = 1
        self.num_workers = 1
        # self.transform =transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, key):
        label = self.labels[key]
        img = self.imgs[key]
        return (img, label)

    def __len__(self):
        return len(self.labels)

def normalize(image):
    img = transforms.ToTensor()(image)
    mn, std = img.mean([1,2]), img.std([1,2])
    transform_norm = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mn, std)
    ])
    tr_img = transform_norm(image)
    return tr_img

"""Data Loading and normalization"""
if __name__ == "__main__":
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
    print(train_no)

    dataset = CustomDataset(images, labels)
    train_data, test_data = torch.utils.data.random_split(dataset, [train_no, test_no], generator=torch.Generator().manual_seed(42))


    #Split into train and test
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=round(train_no/5), shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=round(test_no/5), shuffle=True)
    torch.save(trainloader, '/home/emily/GreenerDNNs/activations/trainloader.pth')
    torch.save(testloader, '/home/emily/GreenerDNNs/activations/testloader.pth')

    # for data, target in trainloader:
    #     print("One")
    #     print(data.shape)
    print("Dataloader created.\n")