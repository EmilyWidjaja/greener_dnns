from timing_class import measure_energy
from convNets.convKernels.models import FashionConv
from tqdm import tqdm
import os

class measure_chan_energy(measure_energy):
    def __init__(self, device_number, times, trials, path, sampling_period, momentum, lr, exp, debug, attributes=''):
        super(measure_chan_energy, self).__init__(device_number, times, trials, path, sampling_period, momentum, lr, exp, debug, attributes)
        return

    def load_model(self, ker, out_channels1, default=False):
        net_obj = FashionConv(ker, out_channels1, self.lr, self.momentum, self.device_number, experiment_name=self.exp, path=self.path, fileformat=self.exp)
        if default:
            self.default_net = net_obj
        return net_obj

    def main(self, kers, out_channels1, train_path, test_path, warm_up_times=0):
        #Warm-up
        self.set_device()
        net_obj = self.load_model(kers, 16, default=True) #Just reminder to set a default!
        self.load_infer_data(train_path, test_path)
        self.instantiate_model(net_obj)
        self.warm_up(net_obj, warm_up_times)

        #Take readings
        if self.debug and len(out_channels1) >= 4:
            out_channels1 = out_channels1[0:2] + out_channels1[-2::]
        for chan in tqdm(out_channels1):
            for trial in range(0, self.trials):
                net_obj = self.load_model(kers, chan)
                self.instantiate_model(net_obj)
                command_str = self.define_command(trial, net_obj)
                self.test(command_str, net_obj)

        print('Timings complete.')
        os.system('tput bel')
        return