from enviroment import WarehouseEnv
from store import Store
from agent import Agent


NUM_DAYS = 500
NUM_EPISODES = 500
""" 
SETUP ENVIROMENT
"""
env = WarehouseEnv(
    max_age=6,
    n_days=NUM_DAYS
    )
env.addStore(
    Store(
        avg=8,
        std=5,
        max_age=6)) 
env.addStore(
    Store(
        avg=13,
        std=5,
        max_age=6))
env.addStore(
    Store(
        avg=20,
        std=5,
        max_age=6))
env.setup_spaces()

"""
SETUP AGENT
"""
scores = []
agent = Agent(
    state_size=env.state_size,
    action_size=env.maxorder,
    learn_rate=     0.001,# NN Learning Rate
    gamma=          0.6, # Long Term Consideration Factor
    epsilon_decay=  0.996,# Epsilon Exponential Decay Factor
    epsilon_min=    0.01,#  Min value for Epsilon
    temperature=    5,#     SoftMax Value Equaliser
    batch_size=     5,#     Num of Training Samples at Once
    memory_size=    100,#   Num of Training Samples in Memory
    target_update_rate=50,# Steps between Target Copy
    policy_save_rate=20,#   Episodes between Saving Policy
)
"""
LOOP
"""
for episode in range(1,NUM_EPISODES+1):
    score = 0
    done = False
    step = 0
    state = env.reset()
    error = False
    while not done:
        """ 
        1. Choose action
        2. Pass action to enviroment
        3. Recieve new state and reward from enviroment
        4. Save observation and reward
        5. Update Agent network
        """
        action = agent.choose_action(state)
        next_state, reward, done, error = env.step(action)

        if error:break
        #use the cumulated reward as an overall score for this episode
        score += reward

        agent.memory.add_transition(transition=(state, action, reward, next_state))
            
        if step % 200 == 0:
            agent.epsilon = max(agent.epsilon*agent.epsilon_decay, agent.epsilon_min)
        
        # We pass batches of observations to the agent
        # so we need at least one batch amount of observations
        if len(agent.memory) > agent.batch_size:
            agent.update()
        
        state = next_state
        step += 1
    if error:break
    scores.append(score)
    
    if episode % agent.policy_save_rate == 0:
        agent.save_model(f"{episode}_({score})_warehouse_agent")
    print(f"Episode: {episode},\tSc: {score},\tAvg Rw/D: {score//NUM_DAYS}\tEps: {agent.epsilon:.3f}")

with open('models\\scores.txt', 'w') as f:
    f.write("\n".join(map(str, scores)))

import matplotlib.pyplot as plt

plt.plot(range(len(scores)), scores)
plt.grid(True)
plt.show()
