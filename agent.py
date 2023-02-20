import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os

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
        
    
    def choose_action(self,state):
        rnd = np.random.random()
        q_values = self.Q_policy(state)
        if rnd < 1 - self.epsilon:
            action = torch.argmax(q_values).item()
        else:
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

        # pass through network
        Q_values = self.Q_policy(states).to(self.Q_policy.device)
        Q_next_values = self.Q_target(next_states).to(self.Q_policy.device)
        
        # loss calculation
        max_q_indexes = torch.argmax(Q_values,dim=1).to(self.Q_policy.device)
        
        # (BELLMANN-EQUATION)
        Q_targets = Q_values.clone()
        Q_targets[np.arange(self.batch_size),max_q_indexes] = rewards + self.gamma*torch.max(Q_next_values[1])
        
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
