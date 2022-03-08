import torch
import os
import csv
from torchvision import transforms
import torchvision
from timing_class import measure_energy
from tqdm import tqdm
import subprocess

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

def write_clocks(path, mem, gr):
    filepath = os.path.join(path, 'current_clocks')
    with open(os.path.join(filepath, 'mem.txt'), 'w') as f:
        f.write('{}'.format(mem))
    with open(os.path.join(filepath, 'gr.txt'), 'w') as f:
        f.write('{}'.format(gr))
    return

def save_accuracy(MEM_CLOCK, GR_CLOCK, trial, accuracy):
    debug = True
    if 'accuracy' not in os.listdir(os.getcwd()): os.system('mkdir accuracy')
    with open('./accuracy/{}-{}_{}.csv'.format(MEM_CLOCK, GR_CLOCK, trial), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([accuracy])
    if debug == True:
        print("data saved successfully.")
    return

def get_supported_clocks(filepath, header=True):
    with open(filepath, "r") as f:
        csv_reader = csv.reader(f)
        clocks = []
        for row in csv_reader:
            if header == True:
                header = False
                continue
            clocks.append(int(row[0]))
    return clocks

def normalize(image):
    img = transforms.ToTensor()(image)
    mn, std = img.mean([1,2]), img.std([1,2])
    transform_norm = transforms.Compose([
        transforms.Resize((50,50)),
        transforms.ToTensor(),
        transforms.Normalize(mn, std)
    ])
    tr_img = transform_norm(image)
    return tr_img


class measure_DVFS_energy(measure_energy):
    def __init__(self, device_number, times, trials, path, sampling_period, momentum, lr, exp, debug, attributes=''):
        super(measure_DVFS_energy, self).__init__(device_number, times, trials, path, sampling_period, momentum, lr, exp, debug, attributes)
        return

    def load_model(self):
        net_obj = torchvision.models.resnet18(pretrained=True)
        net_obj.eval()
        net_obj = net_obj.to(self.device)
        return net_obj

    def warm_up(self, net_obj, times, initial=False):
        print("Warming up!")
        for i in range(times):
            net_obj(self.data)
        if initial == True:
            print("Warm up complete. Temperature is: ")
            os.system('nvidia-smi -q -d temperature')
            os.system('tput bel')
        return

    def define_command(self, trial, mem, gr):
        if self.debug == True:
            command_str = ["nvidia-smi", "--query-gpu=timestamp,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,temperature.gpu,memory.used",
                    "--format=csv,nounits,noheader", "-i", str(self.device_number), "-f", os.path.join(self.path, "energy/temp/{}-{}_{}.txt".format(mem, gr, trial)), "-lms", str(self.period)]
        else:
            command_str = ["nvidia-smi", "--query-gpu=timestamp,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,temperature.gpu,memory.used",
                    "--format=csv,nounits,noheader", "-i", str(self.device_number), "-f", os.path.join(self.path, "energy/{}/{}-{}_{}.txt".format(self.path, self.exp, mem, gr, trial)), "-lms", str(self.period)]
        return command_str

    def test(self, command_str, net_obj, times=50):
    #Run some times to flush for times
        with torch.no_grad():
            #start nvidia-smi in another shell
            proc = subprocess.Popen(command_str)
            for i in range(times):
                output = net_obj(self.data)
                pred = output.data.max(1, keepdim=True)[1]
            proc.terminate()    #Stop nvidia-smi

            return
    
    def main(self, clocks, dataloader, warm_up_times=0):
        #Setup
        self.set_device()
        net_obj = self.load_model() 

        data, target = next(iter(dataloader))
        print("Moving data to device...")
        self.data = data.to(self.device)
        print("Data moved")
        # target = target.to(self.device)
        # print("Target moved.")

        print('Warming up...')
        self.warm_up(net_obj, warm_up_times, initial=True)

        #Take readings
        if self.debug:
            clocks = [(7601, 1110)]
        
        for mem, gr in tqdm(clocks):
            #Compile and change clocks
            write_clocks(self.path, mem, gr)
            command = 'sudo ./change_clocks'
            print(command, '\n\n')
            os.system(command)

            for trial in range(0, self.trials):
                self.warm_up(net_obj, warm_up_times)
                command_str = self.define_command(trial, mem, gr)
                self.test(command_str, net_obj)

        print('Timings complete.')
        os.system('tput bel')
        return
