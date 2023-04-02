import torch
import torch.nn as nn
from torch.autograd import Variable

class RQNetwork(nn.Module):
    def __init__(self,state_size,action_size,lr):
        super(RQNetwork,self).__init__()
        self.state_size = state_size
        
        self.lstm = nn.LSTM(state_size, state_size)
        self.fc1 = nn.Linear(state_size,(action_size+action_size)//2)
        self.fc2 = nn.Linear((action_size+action_size)//2,action_size)
        
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.h_0 = Variable(torch.zeros(1, self.state_size).to(self.device))
        self.c_0 = Variable(torch.zeros(1, self.state_size).to(self.device))
        
    def forward(self,x, learn=False):
        """
        Learning mode separates the simulation from the training
            hidden state and cell state during training is 
            temporarily stored,
            rather then re-used like in the real simulation
        """
        x = torch.Tensor(x).to(self.device)
        if learn:
            h_0 = Variable(torch.zeros(1, self.state_size).to(self.device))

            c_0 = Variable(torch.zeros(1, self.state_size).to(self.device))

        else:
            h_0 = self.h_0
            c_0 = self.c_0

        output, (h_0,c_0) = self.lstm(x, (h_0, c_0))
        output = self.fc1(torch.relu(output[-1]))
        output = self.fc2(torch.relu(output))
        
        if not learn:
            self.h_0 = h_0
            self.c_0 = c_0
        
        return output

class QNetwork(nn.Module):
    def __init__(self,state_size,action_size,lr):
        super(QNetwork,self).__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(state_size,2*state_size),
            nn.ReLU(),
            nn.Linear(2*state_size,2*state_size),
            nn.ReLU(),
            nn.Linear(2*state_size,state_size),
            nn.ReLU(),
            nn.Linear(state_size,(state_size+action_size)//2),
            nn.ReLU(),
            nn.Linear((state_size+action_size)//2,action_size))
        
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self,x):
        x = torch.Tensor(x).to(self.device)

        x = self.linear(x)
        return x
