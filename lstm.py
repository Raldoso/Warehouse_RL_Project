import torch
import torch.nn as nn

from torch.autograd import Variable

class QNetwork(nn.Module):
    def __init__(self,state_size,action_size,lr):
        self.state_size = state_size
        super(QNetwork,self).__init__()
        
        self.lstm = nn.LSTM(state_size, state_size)
        self.fc1 = nn.Linear(state_size,(action_size+action_size)//2)
        self.fc2 = nn.Linear((action_size+action_size)//2,action_size)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self,x):
        x = torch.Tensor(x).to(self.device)
        h_0 = Variable(torch.zeros(1, self.state_size))
        c_0 = Variable(torch.zeros(1, self.state_size))

        output, _ = self.lstm(x, (h_0, c_0))
        output = self.fc1(torch.relu(output[-1]))
        output = self.fc2(torch.relu(output))
        return output
    
x = QNetwork(2, 3, 0.001)
input = torch.rand(3, 2) #sequence, features
output = x.forward(input)
print(output)

    