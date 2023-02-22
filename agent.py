import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os

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
            nn.Linear(state_size,state_size),
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

class StateMemory():
    def __init__(self,capacity,batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = []

    def add_transition(self,transition):
        # save new transition
        # delete old transition if full to keep fixed capacity
        if len(self) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)
        
    def sample(self):
        # return 1 batch of transitions
        start_index = np.random.randint(0,len(self)-self.batch_size+1)
        return self.memory[start_index:start_index+self.batch_size]

    def __len__(self):
        return len(self.memory)
   
class Agent():
    def __init__(self,
                 state_size,
                 action_size,
                 learn_rate,
                 gamma,
                 epsilon_decay,
                 epsilon_min,
                 temperature,
                 batch_size,
                 memory_size,
                 target_update_rate,
                 policy_save_rate,
                ):
        # ENVIROMENT PARAMETERS
        self.state_size = state_size
        self.action_size = action_size

        # NETWORK LEARNING PARAMETERS
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.temperature = temperature
        self.target_update_rate = target_update_rate
        
        self.policy_save_rate = policy_save_rate

        # MEMORY PARAMETERS
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = StateMemory(self.memory_size,self.batch_size)
        
        self.step = 0
        self.Q_policy = QNetwork(state_size,action_size,self.learn_rate)
        self.Q_target = QNetwork(state_size,action_size,self.learn_rate)
    
    def choose_action(self,state,simulate=False):
        rnd = np.random.random()
        q_values = self.Q_policy.forward(torch.Tensor(state).view(1,self.state_size)).to(self.Q_policy.device) # add batch dimension for the NN

        if rnd < 1 - self.epsilon or simulate:
            print("max")
            action = torch.argmax(q_values).item()
        else:
            print("rand")
            action = np.random.choice(np.arange(self.action_size))
        return action
        
    def save_model(self,name):
        if not os.path.exists('models'):
            os.mkdir("models")
        torch.save(self.Q_policy.state_dict(), f"models\\{name}.pth")

    def load_model(self,path):
        self.Q_policy.load_state_dict(torch.load(path))

    def update(self):
        self.Q_policy.optimizer.zero_grad()
        
        state_batch = np.array(self.memory.sample(),dtype=object)

        # data from batch
        states = torch.Tensor(list(state_batch[:,0])).to(self.Q_policy.device)
        next_states = torch.Tensor(list(state_batch[:,3])).to(self.Q_policy.device)
        rewards = torch.Tensor(list(state_batch[:,2])).to(self.Q_policy.device)

        # pass through network in learning mode
        Q_values = self.Q_policy.forward(states).to(self.Q_policy.device)
        Q_next_values = self.Q_target.forward(next_states).to(self.Q_policy.device)
        
        # loss calculation
        max_q_index = torch.argmax(Q_values,dim=1).to(self.Q_policy.device)
        
        # (BELLMANN-EQUATION)
        Q_targets = Q_values.clone()
        Q_targets[np.arange(self.batch_size),max_q_index] = rewards + self.gamma*torch.max(Q_next_values[1])
        
        loss = self.Q_policy.loss(Q_targets,Q_values).to(self.Q_policy.device)
        loss.backward()
        self.Q_policy.optimizer.step()
        
        # network parameters update
        if self.step % self.target_update_rate == 0:
            # print("Copy target network")
            self.Q_target.load_state_dict(self.Q_policy.state_dict())
        
        #self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        self.step += 1
if __name__ == "__main__":
    # print(torch.rand(3,1,5))
    x = torch.rand(3,2)
    y = torch.rand(3,2)
    print(x)
    print(y)
    # print(y.split(1,dim=1))
    # [print(i) for i in y.split(1,dim=0)]
    # print(torch.cat((x,y),dim=1))
    for i in range(y.size()[0]):
        print(y[i])
