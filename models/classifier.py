from __future__ import print_function

import torch.nn as nn
import torch

#########################################
# ===== Classifiers ===== #
#########################################

class LinearClassifier(nn.Module):

    def __init__(self, dim_in, n_label=10):
        super(LinearClassifier, self).__init__()

        self.net = nn.Linear(dim_in, n_label)

    def forward(self, x):
        return self.net(x)


class NonLinearClassifier(nn.Module):

    def __init__(self, dim_in, n_label=10, p=0.1):
        super(NonLinearClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, 200),
            nn.Dropout(p=p),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Linear(200, n_label),
        )

    def forward(self, x):
        return self.net(x)

class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class MLP(nn.Module):
    def __init__(self, hidden_size=100, num_layers=1, input_size=200):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(input_size, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        # self.output_layer = nn.Linear(hidden_size, 1)
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        # return torch.sigmoid(x)
        
        # 0~2
        # k1 = 2/(max(x[:,0])-min(x[:,0]))
        # k2 = 2/(max(x[:,1])-min(x[:,1]))
        # t1 = (k1 * (x[:,0]-min(x[:,0]))).view(-1, 1)
        # t2 = (k2 * (x[:,1]-min(x[:,1]))).view(-1, 1)
        
        # 0.5~1.5
        k1 = 1/(max(x[:,0])-min(x[:,0]))
        k2 = 1/(max(x[:,1])-min(x[:,1]))
        t1 = (0.5 + k1 * (x[:,0]-min(x[:,0]))).view(-1, 1)
        t2 = (0.5 + k2 * (x[:,1]-min(x[:,1]))).view(-1, 1)
        
        return torch.cat((t1,t2), 1)