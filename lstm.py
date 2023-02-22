import torch
import torch.nn as nn

from torch.autograd import Variable

class QNetwork(nn.Module):
    def __init__(self,state_size,action_size,lr):
        super(QNetwork,self).__init__()
        self.state_size = state_size
        
        self.lstm = nn.LSTM(state_size, state_size)
        self.fc1 = nn.Linear(state_size,(action_size+action_size)//2)
        self.fc2 = nn.Linear((action_size+action_size)//2,action_size)
        
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.to(self.device)
        self.h_0 = Variable(torch.zeros(1, self.state_size))
        self.c_0 = Variable(torch.zeros(1, self.state_size))
        
    def forward(self,x,learn=False):

        x = torch.Tensor(x)

        if learn:
            h_0 = Variable(torch.zeros(1, self.state_size))
            c_0 = Variable(torch.zeros(1, self.state_size))
        else:
            h_0 = self.h_0
            c_0 = self.c_0

        output, (h_0,c_0) = self.lstm(x, (h_0, c_0))
        output = self.fc1(torch.relu(output[-1]))
        output = self.fc2(torch.relu(output))
        
        if not learn:
            self.h_0 = h_0
            self.c_0 = c_0
        
        return output, h_0,c_0
if __name__ == "__main__":
    #torch.manual_seed(1) 
    x = QNetwork(2, 3, 0.001)
    input = torch.rand(3,2) #sequence, features
    #input1 = torch.rand(2).view(1,2)
    input1 = torch.rand(2)
    
    #output, h,c = x.forward(input1,learn=True)
    net = nn.Linear(2,3)
    out = net(input)
    print(out)
    print(torch.max(out[1]))
    print(torch.Tensor([1,2])+1)
    


    