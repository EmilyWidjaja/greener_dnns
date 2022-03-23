import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torch.optim as optim
from default_variables import batch_size_train, data_size
import os
import csv
"""
Classes to override:
class Net(nn.Module)

class Net
class FashionVal():
    def __init__(self, no_layers, no_nodes, lr, momentum, path):
    def instantiate_net(self, vars):

        """
#Set Network
class Net(nn.Module):
    def __init__(self, no_layers, no_nodes):
        #instantiates DNN with no_layers fully connected layers, a flattened input layer and a 10-node output layer
        super(Net, self).__init__()
        

        n = no_nodes
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features=28*28, out_features=n))
        self.d0 = nn.Linear(in_features=28*28, out_features=50)
        for i in range(0, no_layers):
            self.layers.append(nn.Linear(in_features=n, out_features=n))
        self.layers.append(nn.Linear(in_features=n, out_features=10))
        print("Model Architecture: {} hidden layers of {} nodes.".format(no_layers, n))

    def forward(self, x): 
        x = x.reshape(-1, 28*28).cuda()
        for layer in self.layers:
            x = layer(x)
        return F.log_softmax(x)

class FashionVal(object):
    def __init__(self, lr, momentum, device_number, experiment_name, path):
        #Initialize device
        torch.cuda.set_device(device_number)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(torch.cuda.get_device_name(self.device), ' device used.')
        if self.device == 'cpu':
            raise SystemExit(0)

        #set constants
        self.path = path
        self.fileformat = ''
        self.exp = experiment_name
        self.learning_rate = lr
        self.momentum = momentum
        self.prev_epochs = 0
        self.batch_size_train = batch_size_train
        self.data_size = data_size

        #Set random seed for same behaviour upon retraining
        random_seed = 1
        torch.backends.cudnn.enabled = False
        torch.manual_seed(random_seed)


        #Check needed folders for saving are present (changing these names requires changing keys in instantiate model, train_model, save_training_data)
        self.paths_dict = {'results': '', 'train_data': '', 'energy':''}
        self.initialize_paths()
        
        return

    def set_batch_size_train(self, batch_size_train, data_size):
        self.batch_size_train = batch_size_train
        self.data_size = data_size
        return

#HELPER METHODS
    def initialize_paths(self):
        #Checks all paths exist
        self.paths_set = True
        for key in self.paths_dict.keys():
            path = os.path.join(self.path, key, self.exp)
            self.paths_dict[key] = path
            self.verify_path(path)
        if self.paths_set == False:
            print('Path not able to be verified. Terminating program...')
            raise SystemExit(0)
        return
    
    def print_memory(self, device_number = 1):
        #Prints memory reserved & allocated for CUDA devices. Should be similar (otherwise clear cache)
        t = torch.cuda.get_device_properties(device_number).total_memory
        r = torch.cuda.memory_reserved(device_number)
        a = torch.cuda.memory_allocated(device_number)
        print('Total Memory: {}\nReserved: {:.2f}%\nAllocated: {:.2f}%'.format(t, float(r)/t*100, float(a)/t*100))
        return
    
    def error_log(self, test_loss):
        #Just a quick error log to know which ones didn't train for overnight runs
        info = "Training error hit: {} for\t {}".format(test_loss, self.fileformat)
        with open(os.path.join(self.paths_dict['results'], 'error_log.txt'), 'a') as f:
            f.write(info+'\n')
        return

    def verify_path(self, path):
        #Split into parts, and validate all parts
        folders = path.split('/')

        #Check if last one is file or path
        if '.' in folders[-1]:
            folders = folders[:-1]

        tested = "/"
        for folder in folders:
            tested = os.path.join(tested, folder)
            if os.path.exists(tested) != True:
                print('Folder does not exist {}. Making folder...'.format(tested))
                os.system('mkdir {}'.format(tested))
        
        # print('{} exists: {}'.format(path, os.path.exists(path)))
        if os.path.exists(path) == False:
            self.paths_set = False
        return 

    def save_training_data(self, epochs, best_acc, best_test_loss):
        with open(os.path.join(self.paths_dict['train_data'], '{}.csv'.format(self.fileformat)), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.train_counter)
            writer.writerow(self.train_losses)
            writer.writerow(self.test_counter)
            writer.writerow(self.test_losses)
            writer.writerow([epochs + self.prev_epochs])
            writer.writerow([best_acc])
            writer.writerow([best_test_loss])
        print("data saved as {}".format(self.fileformat))
        return

#TRAINING METHODS
    def load_data(self, train_path, test_path):
        #Loads pre-processed dataloaders
        train_loader = torch.load(train_path)
        test_loader = torch.load(test_path)
        print("Dataloader loaded.")
        return train_loader, test_loader

    def instantiate_net(self):
        #Method to be overwritten
        self.network = Net(self.m, self.n)
        return

    def instantiate_model(self, train_more=False):
        #Instantiates model w/ optimizer with option to continue training
        self.instantiate_net()
        self.network.to(self.device)
        self.train_more = train_more
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=self.momentum)
        if train_more == False:
            #loaded from scratch
            print("Model instantiated.")
            print(summary(self.network, self.data_size, self.batch_size_train))
        else:
            #loaded in continued state
            network_state_dict = torch.load(os.path.join(self.paths_dict['results'], 'model{}.pth'.format(self.fileformat)))
            self.network.load_state_dict(network_state_dict)
            optimizer_state_dict = torch.load(os.path.join(self.paths_dict['results'], 'optimizer{}.pth'.format(self.fileformat)))
            self.optimizer.load_state_dict(optimizer_state_dict)
        return

    def train(self, train_loader, epoch):
        #Training loop for 1 epoch

        self.network.train()
        torch.cuda.empty_cache()
        batches = len(train_loader)
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.network(data)

            loss = F.cross_entropy(output, target)   #find loss between expected and produced y
            torch.cuda.empty_cache()
            loss.backward() #computes derivative of loss using backpropogation
            self.optimizer.step()    #optimizer takes a step based on gradients of parameters
            if batches > 4 and batch_idx % 4 == 0:
                print('Train epoch: {ep}\t [batch {batch_no}/{batches}]\t loss: {loss:.6f}'.format(\
                    ep=self.prev_epochs+epoch+1, batch_no=batch_idx+1, batches=batches, loss=loss.item()))
            self.train_losses.append(loss.item())
            self.train_counter.append(
                (batch_idx*self.batch_size_train) + ((self.prev_epochs+epoch)*len(train_loader.dataset)))
        return

    def test(self, test_loader):
        #Testing loop for one epoch
        self.network.eval()     #turns off dropout layers, batchnorm layers
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.network(data) # probabilities of all the classes (10 probabilities) (1000 x 10)
                test_loss += F.cross_entropy(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]  #selects the class with the maximum probability (1000 x 1)
                # print(pred)
                correct += pred.eq(target.data.view_as(pred)).sum().item() # compares predicted to target & sums 
        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)
        accuracy = 100.*correct/len(test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{}\
            ({corr:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), corr=100.*correct/len(test_loader.dataset)))

        return accuracy, test_loss

    def train_model(self, epochs, train_loader, test_loader):
        #Runs train loop for all epochs

        #Initialize record-keeping depending on if we continue training
        if self.train_more == False:
            self.train_losses = []
            self.train_counter = []
            self.test_losses = []
            self.test_counter = [i*len(train_loader.dataset) for i in range(epochs+1)]
        else:
            with open(os.path.join(self.paths_dict['train_data'], "{}.csv".format(self.fileformat)), "r") as file:
                csv_reader = csv.reader(file)
                lists_from_csv = []
                for row in csv_reader:
                    lists_from_csv.append(row)
            self.train_counter = lists_from_csv[0]
            self.train_losses = lists_from_csv[1]
            self.test_counter = lists_from_csv[2]
            self.test_losses = lists_from_csv[3]
            self.prev_epochs = int(lists_from_csv[4][0])
            new_counts = [i*self.batch_size_train for i in range(self.prev_epochs, self.prev_epochs+epochs+1)]
            self.test_counter.extend(new_counts)
        
        
        print("Start training: ")
        best_acc, best_test_loss = self.test(test_loader)
        torch.save(self.network.state_dict(), os.path.join(self.paths_dict['results'], 'model{}.pth'.format(self.fileformat)))
        torch.save(self.optimizer.state_dict(), os.path.join(self.paths_dict['results'], 'optimizer{}.pth'.format(self.fileformat)))
                
        for epoch in range(epochs):
            self.train(train_loader, epoch)
            accuracy, test_loss = self.test(test_loader)
            if test_loss <  best_test_loss:
                best_test_loss = test_loss
                best_acc = accuracy
                torch.save(self.network.state_dict(), os.path.join(self.paths_dict['results'], 'model{}.pth'.format(self.fileformat)))
                torch.save(self.optimizer.state_dict(), os.path.join(self.paths_dict['results'], 'optimizer{}.pth'.format(self.fileformat)))
                print("Saved new model {}".format(self.fileformat))
            if test_loss > 10000:
                print("Training stopped due to large training loss")
                self.error_log(test_loss)
                os.system('tput bel')
                break
        print("best accuracy: {}\tbest test loss: {}".format(best_acc, best_test_loss))
        return best_acc, best_test_loss

    def main(self, epochs, train_path='./train_loader.py', test_path='./test_loader.py', train_more=False): #main train&testing sequence
        train_loader, test_loader = self.load_data(train_path, test_path)
        self.instantiate_model(train_more)
        best_acc, best_test_loss = self.train_model(epochs, train_loader, test_loader)
        self.save_training_data(epochs, best_acc, best_test_loss)
        return best_acc, best_test_loss

        
