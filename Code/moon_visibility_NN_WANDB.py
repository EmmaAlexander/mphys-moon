import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
#import os
#print(os.getcwd())
import wandb
wandb.login()

#wandb.init(project="moon_visibility_NN_WANDB")

sweep_config = {
    "method": "bayes", # try grid or random
    "metric": {
      "name": "test loss",
      "goal": "minimize"   
    },
    "parameters": {
        "batch_size": {
            'distribution': 'int_uniform',
            #'q': 2,
            'min': 32,
            'max': 256,
        },
        "hidden_size": {
            'distribution': 'int_uniform',
            #'q': 2,
            'min': 4,
            'max': 32,
        },
        "num_epochs": {
            'distribution': 'int_uniform',
            'min': 50,
            'max': 300,
        },
        "learning_rate": {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.05,
        },
        "weight_decay": {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
        },
        "layer_num": {
            'values': [1, 2, 3]
        }
    }
}


PARAMS = {'batch_size': 32, 'hidden_size': 32, 'num_epochs': 200, 'learning_rate': 0.01 , 'weight_decay': 0.01,'layer_num':1} #
METHOD = False
USE_GPU = False
LINUX = False

if USE_GPU:
    #torch.zeros(1).cuda()
    #torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
else:
    device = torch.device('cpu')

icouk_data_file = '..\\Data\\icouk_sighting_data_with_params.csv'
icop_data_file = '..\\Data\\icop_ahmed_2020_sighting_data_with_params.csv'
alrefay_data_file = '..\\Data\\alrefay_2018_sighting_data_with_params.csv'
allawi_data_file = '..\\Data\\schaefer_odeh_allawi_2022_sighting_data_with_params.csv'

if LINUX:
    icouk_data_file = '../Data/icouk_sighting_data_with_params.csv'
    icop_data_file = '../Data/icop_ahmed_2020_sighting_data_with_params.csv'
    alrefay_data_file = '../Data/alrefay_2018_sighting_data_with_params.csv'
    allawi_data_file = '../Data/schaefer_odeh_allawi_2022_sighting_data_with_params.csv'

icouk_data = pd.read_csv(icouk_data_file)
icop_data = pd.read_csv(icop_data_file)
alrefay_data = pd.read_csv(alrefay_data_file)
allawi_data = pd.read_csv(allawi_data_file)

data = pd.concat([icouk_data,icop_data,alrefay_data,allawi_data])

#print(f"Loaded {data.shape[0]} rows")

data = data.drop(["Index","q","W","q'","W'",'Visibility'], axis = 1)

if METHOD: # method and methods columns, will be changed
    data = data.drop('Seen', axis = 1) # replaced by method column
    ptype = [r"Not_seen", r"Seen_eye", r"Seen_binoculars", r"Seen_telescope", r"Seen_ccd"] # CHANGE THIS
else:
    data = data[data["Method"] !="Seen_binoculars"] #DROP BINOCULARS
    data = data[data["Method"] !="Seen_ccd"] #DROP CCD
    data = data[data["Method"] !="Seen_telescope"] #DROP TELESCOPE
    
    data=data.drop(['Method','Methods'], axis = 1)
    # List of label options
    ptype = [r"Seen", r"Not_seen"]

print(f"Selected {data.shape[0]} rows")

variable_list =  data.columns.tolist()
variable_list.remove('Seen')

data['Sunset'] -= data['Date'] # reducing magnitude of date datapoints
data['Moonset'] -= data['Date']
data['Date'] -= 2400000
data['Date'] *= 1/10000

y = np.array(data['Seen'])
X = np.array(data[variable_list])

y[y == 'Seen'] = int(1)
y[y == 'Not_seen'] = int(0)

def sweep_train():  
    config_defaults = PARAMS
    wandb.init(config=config_defaults)  # defaults are over-ridden during the sweep

    batch_size = wandb.config['batch_size']
    hidden_size = wandb.config['hidden_size']
    num_epochs = wandb.config['num_epochs']
    learning_rate = wandb.config['learning_rate']
    weight_decay = wandb.config['weight_decay']
    layer_num = wandb.config['layer_num']

    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, y,test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

    # Define the neural network model
    class CustomNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CustomNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            for i in range(0,layer_num):
                x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
    input_size = len(variable_list)  # Number of input variables
    output_size = 2 # for binary output
    model = CustomNN(input_size, hidden_size, output_size).to(device)

    # Define binary cross-entropy loss and Adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay) #VARIED BY WANDB
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=2, factor=0.9)

    X_train_tensor = torch.from_numpy(X_train).float()
    Y_train_tensor = torch.from_numpy(Y_train.astype("float64")).float()

    X_test_tensor = torch.from_numpy(X_test).float()
    Y_test_tensor = torch.from_numpy(Y_test.astype("float64")).float()

    #data prep
    trainset = TensorDataset(X_train_tensor, Y_train_tensor)
    testset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    def train(model, trainloader, optimiser, device):

        train_loss = 0.0

        model.train()
        for batch_idx, (data, labels) in enumerate(trainloader):
            data, labels = data.to(device), labels.to(device)

            optimiser.zero_grad()

            p_y = model(data)
            loss_criterion = nn.CrossEntropyLoss()
            labels = labels.type(torch.LongTensor) 
            labels = labels.to(device)
            loss = loss_criterion(p_y, labels)
                
            train_loss += loss.item() * data.size(0)

            loss.backward()
            optimiser.step()

        train_loss /= len(trainloader.dataset)
        return train_loss

    def test(model, testloader, device):

        correct = 0
        test_loss = 0.0

        model.eval()
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(testloader):
                data, labels = data.to(device), labels.to(device)

                p_y = model(data)
                loss_criterion = nn.CrossEntropyLoss()
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)
                loss = loss_criterion(p_y, labels)
                    
                test_loss += loss.item() * data.size(0)

                preds = p_y.argmax(dim=1, keepdim=True)
                correct += preds.eq(labels.view_as(preds)).sum().item()

            test_loss /= len(testloader.dataset)
            accuracy = correct / len(testloader.dataset)

        return test_loss, accuracy

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        
        train_loss = train(model, train_loader, optimizer, device)
        test_loss, accuracy = test(model, test_loader, device)
            
        scheduler.step(test_loss)
    ###################################################################################################################
        wandb.log({'accuracy':accuracy,'train loss':train_loss,'test loss':test_loss})
        #results = [epoch, train_loss, test_loss, accuracy]

sweep_id = wandb.sweep(sweep_config, project="moon_visibility_NN_WANDB")
wandb.agent(sweep_id, sweep_train)
wandb.finish()