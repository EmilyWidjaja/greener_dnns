
"""Due to issues with the A5000, where it could not load the dataloader, so dataprocessing is done here.

Procedure:
1. Load & process data (this should just be torch.load(dataloader), but that doesn't work somehow)
2. Load available clock combinations
3. Define processor that loads the pretrained ResNet50 model
4. Iterate through available clock combinations & take 5 energy readings each
    - For each clock combination, compile a CUDA script that changes clock frequencies & run it.
    (clock combination written to a text file, compile, then CUDA script reads the text file at runtime - there's probably a better way to do this)
"""

#%% initiate
import torch
import os
import sys

parentdir = '/home/emily/GreenerDNNs'
sys.path.append(parentdir) 

from PIL import Image
import numpy as np
from tqdm import tqdm
from DVFS_helpers import *
from alexnet_vars import *

if __name__ == "__main__":
    """Data Loading and normalization"""
    actual_images = False

    if actual_images == True:
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
        # Remove B&W images

        bw = []
        idx = 0
        while idx < len(images):
            #Find indexes of B&W images and remove
            if images[idx].size()[0] == 1:  #If first dimension has 1 channel - i.e. B&W
                bw.append(idx)
                images.pop(idx)
                data_dict[decoder[labels[idx]]] -= 1    #Update dictionary with total count
                labels.pop(idx)
                continue
            idx += 1
        
        print(len(bw), " B&W images removed.")
        print(len(labels), " images remaining.")
        labels = np.array(labels)

    else:
        #Create rubbish data
        size = (1200, 3, 224, 224)
        images = torch.rand(size, dtype=torch.float32)
        labels = np.random.randint(0, 9, size=size[0], dtype=np.int64)

    #Create dataloader
    print("Creating dataloader...")
    print('Images size: ', len(images), type(images[0]), images[0].size())
    print('Labels size: ', len(labels), type(labels[0]))
    # input('continue?')
    dataset = CustomDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(labels), shuffle=True)
    print("Dataloader created.\n")


    """START TIMINGS"""
    #----------SWITCHES----------------
    #Check GPU
    print(f"""
        Device Number = {device_number},
        {experiment_name}: {trials} trials, {times} times,
        Sampling period {sampling_period}ms,
        Debug: {debug}
        Saved on path {path} 
        """)
    mes = measure_DVFS_energy(device_number, times, trials, path, sampling_period, 0, 0, experiment_name, debug, attributes)

    #Load model & Data
    os.system('nvcc -o change_clocks nvml_run.cu -I/usr/local/cuda-11.6/targets/x86_64-linux/include -L/usr/local/cuda/lib64 -lnvidia-ml')
  
    torch.cuda.empty_cache()
    mes.main(old_path, clocks, dataloader, model='alexnet', warm_up_times=warm_up_times)

    #Reset
    print('Resetting application clocks...')
    os.system('sudo ./reset_clocks')
    os.system("tput bel")