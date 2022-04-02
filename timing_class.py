
# %% initiate
from abc import abstractmethod
import torch
import os
from statistics import mean, stdev
import subprocess
from tqdm import tqdm
import pandas as pd
import datetime
torch.cuda.empty_cache()
"""Method version of timing class
Methods to be overwritten:
- load_model(self, m, n) -> returns net_obj (All of it for different formats of the fashion vals)
- main (since it calls load_model)"""

#Check GPU exists
class measure_energy(object):
    def __init__(self, device_number, times, trials, path, sampling_period, momentum, lr, exp, debug, attributes=''):
        self.device_number = device_number
        self.times = times
        self.trials = trials
        self.path = path
        self.period = sampling_period
        self.momentum = momentum
        self.lr = lr
        self.exp = exp
        self.debug = debug

        if attributes=='':
            self.attributes = 'timestamp,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,temperature.gpu,memory.used,pstate'
        else:
            self.attributes = attributes
        #Additional variables that should be overwritten
        return

    def set_device(self):
        torch.cuda.set_device(self.device_number)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device, ' device used.')
        if self.device == 'cpu':
            print('GPU Device not found')
            raise SystemExit(0)

    @abstractmethod
    def load_model(self, m=0, n=0, default=False):
        pass

    def load_infer_data(self, train_path, test_path):
        train_loader, test_loader = self.default_net.load_data(train_path, test_path)
        train_data, _ = next(iter(train_loader))
        test_data, _ = next(iter(test_loader))
        train_data.to(self.device)
        test_data.to(self.device)

        data = torch.cat((train_data, test_data), dim=0)
        self.data = data.to(self.device)
        #Check
        print('Data is on cuda: {}'.format(self.data.is_cuda))
        return 

    def instantiate_model(self, net_obj):
        net_obj.instantiate_model(train_more=True)
        net_obj.network = net_obj.network.to(self.device)
        print('Model loaded: {}'.format(net_obj.fileformat))
        return

    def warm_up(self, net_obj, warm_up_times=0):
        if warm_up_times == 0:
            warm_up_times = self.times
        print("Warming up!")
        for i in range(warm_up_times):
            net_obj.network(self.data)
        print("Warm up complete. Temperature is: ")
        os.system('nvidia-smi -q -d temperature')
        return net_obj

    def test(self, command_str, net_obj):
        with torch.no_grad():
            #start nvidia-smi in another shell
            proc = subprocess.Popen(command_str)
            for i in range(self.times):
                output = net_obj.network(self.data)
                pred = output.data.max(1, keepdim=True)[1]
            proc.terminate()    #Stop nvidia-smi
        return
    
    def define_command(self, trial, net_obj, timing_name):
        if self.debug == True:
            net_obj.verify_path(os.path.join(self.path, 'energy', '{}_test'.format(self.exp)))
            command_str = ["nvidia-smi", "--query-gpu={}".format(self.attributes), "--format=csv,nounits,noheader", "-i", \
                str(self.device_number), "-f", "{}/energy/{}_test/{}_{}.txt".format(self.path, timing_name, net_obj.fileformat, trial), "-lms", str(self.period)]
        else:
            command_str = ["nvidia-smi", "--query-gpu={}".format(self.attributes), "--format=csv,nounits,noheader", "-i", \
                str(self.device_number), "-f", "{}/energy/{}/{}_{}.txt".format(self.path,timing_name, net_obj.fileformat, trial), "-lms", str(self.period)]
            print(command_str)
        return command_str

    def sampling_iterator(self, timing_name):
        folder_path = os.path.join(self.path, 'energy', timing_name + '_test')
        files = os.listdir(folder_path)
        checks = []
        for file in files:
            sampling_periods = self.check_sampling(os.path.join(folder_path, file))
            print(f"""{file}
            Number of samples: {len(sampling_periods)}\t[Max, min]: {max(sampling_periods)}, min: {min(sampling_periods)}
            [Mean, std]: {mean(sampling_periods):0.4f}ns, {stdev(sampling_periods):0.4f}ns\n""")
            if len(sampling_periods) < 200:
                checks.append((file, len(sampling_periods)))
        if len(checks) != 0:
            print('------------WARNING--------------')
            for file, samples in checks:
                print(f"{file} has {samples} samples")
        return

    def check_sampling(self, file):
        power_data = pd.read_csv(file)
        cols = ['timestamp', 'power', 'gpu_clock', 'mem_clock', 'utilization.gpu', 'utilization.memory']
        self.len_columns = len(power_data.columns) - 6
        if len(power_data.columns) == 6:
            pass
        if len(power_data.columns) >= 7:
            cols.append('temperature')
        if len(power_data.columns) >= 8:
            cols.append('memory_used')
        if len(power_data.columns) >= 9:
            cols.append('pstate')
        power_data.columns = cols
        # Iterate through power_data and multiply power with timestamp
        sampling_periods = []
        total_power = 0
        time_format = '%Y/%m/%d %H:%M:%S.%f'
        prev = False
        for idx, row in power_data.iterrows():
            # Turn into timestamp
            if row.isnull().values.any():
                continue
            curr = datetime.datetime.strptime(row['timestamp'], time_format)
            if prev != False:
                diff = curr-prev
                power = power_data['power'][idx-1] * diff.microseconds * 1e-6
                total_power += power
                sampling_periods.append(diff.microseconds * 1e-6)
            prev = curr
        if power_data.empty:
            print('Dataframe empty')
            raise SystemExit(0)
        return sampling_periods



    def main(self, MN, train_path, test_path, timing_name, warm_up_times=0):
        #Warm-up
        self.set_device()
        net_obj = self.load_model(106, 106, default=True)
        self.instantiate_model(net_obj)
        self.load_infer_data(train_path, test_path)
        self.warm_up(net_obj, warm_up_times)

        #Take readings
        for m, n in tqdm(MN):
            for trial in range(0, self.trials):
                net_obj = self.load_model(m, n)
                self.instantiate_model(net_obj)
                command_str = self.define_command(trial, net_obj, timing_name)
                self.test(command_str, net_obj)

        print('Timings complete.')
        os.system('tput bel')
        return
