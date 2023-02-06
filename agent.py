import torch
import torch.nn as nn
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self,state_size,action_size,lr):
        super(QNetwork,self).__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(state_size,state_size),
            nn.ReLU(),
            nn.Linear(state_size,(action_size+action_size)//2),
            nn.ReLU(),
            nn.Linear((action_size+action_size)//2,action_size))
        
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
        self.memory = deque(maxlen=capacity)

    def add_transition(self,state,action,reward,next_state):
        # save new transition
        # delete old transition if full to keep fixed capacity
        if len(self) >= self.capacity:
            self.memory.popleft()
        self.memory.append((state,action,reward,next_state))
        
    def sample(self):
        # return 1 batch of transitions
        ...
    def __len__(self):
        return len(self.memory)

        
class Agent():
    def __init__(self,
                 state_size,
                 action_size,
                 learn_rate,
                 gamma,
                 epsilon,
                 temperature,
                 batch_size,
                 memory_size,
                ):
        self.state_size = state_size
        self.action_size = action_size
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.temperature = temperature
        self.batch_size = batch_size
        self.memory_size = memory_size
        
        self.Q_policy = QNetwork(state_size,action_size,self.learn_rate)
        self.Q_target = QNetwork(state_size,action_size,self.learn_rate)
        
        self.memory = StateMemory(self.memory_size,self.batch_size)
    
    def choose_action(self,state):
        rnd = torch.rand()
        q_values = self.Q_policy(state)
        if rnd < 1 - self.epsilon:
            action = torch.argmax(q_values).item()
        else:
            action = np.random.choice(np.arange(self.action_size))
        return action
        
    def save_model(self):
        torch.save(self.Q_policy.state_dict(), "warehouse_agent.pth")

    def update(self):
        
        ...
