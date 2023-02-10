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
                ):
        # ENVIROMENT PARAMETERS
        self.state_size = state_size
        self.action_size = action_size

        # NETWORK LEARNING PARAMETERS
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.temperature = temperature
        self.target_update_rate = target_update_rate

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
        
    def save_model(self):
        torch.save(self.Q_policy.state_dict(), "warehouse_agent.pth")

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
        
        # BELLMANN-EQUATION
        Q_targets = Q_values.clone()
        Q_targets[np.arange(self.batch_size),max_q_indexes] = rewards + self.gamma*torch.max(Q_next_values[1])
        
        loss = self.Q_policy.loss(Q_targets,Q_values).to(self.Q_policy.device)
        loss.backward()
        self.Q_policy.optimizer.step()
        
        # network paramters update
        if self.step % self.target_update_rate == 0:
            # print("Copy target network")
            self.Q_target.load_state_dict(self.Q_policy.state_dict())
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        self.step += 1
        
     
if __name__ == "__main__":
            
    # x = np.array([[[4,10,24],2,3,1],[[1,4,1],4,7,8]],dtype=object)
    # x = list(x[:,0])
    # x = torch.Tensor(x)
    # print(x)

    # x = x.view(-1)
    # print(x)

    # x = [[[4,10,24],2,3,1],[[1,4,1],4,7,8]]
    # x = torch.Tensor(list(np.array(x)[:,0])).view(-1)
    # # x = np.vstack(x).astype(np.float32)
    # x = list(x)
    # print(x)
    # torch.seed(2)
    # print(torch.Tensor(x).view(-1))
    torch.manual_seed(2)

    inp = torch.Tensor([[2,3],[7,6]])

    model = nn.Linear(2,4)
    x = model.forward(torch.Tensor(inp))
    print(x)
    print(torch.argmax(x,dim=1))
    
    x = np.array([1,3,1])
    y = []
    y.extend(x)
    print(y)
    
    