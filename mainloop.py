from enviroment import WarehouseEnv, Store
from agent import Agent

""" 
SETUP ENVIROMENT
"""
env = WarehouseEnv(
    max_age=6,
    n_days=500
    )
env.addStore(
    Store(
        avg_range=[8],
        std_range=[5],
        max_age=6)) 
env.addStore(
    Store(
        avg_range=[13],
        std_range=[5],
        max_age=6))
env.addStore(
    Store(
        avg_range=[20],
        std_range=[5],
        max_age=6))
env.setup_spaces()

"""
SETUP AGENT
"""
NUM_EPISODES = 500
scores = []
agent = Agent(
    state_size=env.state_size,
    action_size=env.maxorder,
    learn_rate=     0.001,
    gamma=          0.99,
    epsilon_decay=  0.996,
    epsilon_min=    0.05,
    temperature=    5,#not important atm
    batch_size=     3,
    memory_size=    100,
    target_update_rate=50,
    policy_save_rate=20,
)
""" 
LOOP
"""
for episode in range(NUM_EPISODES):
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
            
        if episode % agent.policy_save_rate == 0:
            agent.save_model(f"{episode}_({score})_warehouse_agent")
        if step % 100 == 0:
            agent.epsilon = max(agent.epsilon*agent.epsilon_decay, agent.epsilon_min)
        
        # We pass batches of observations to the agent
        # so we need at least one batch amount of observations
        if len(agent.memory) > agent.batch_size:
            agent.update()
        
        state = next_state
        step += 1
    if error:break
    scores.append(score)
    
    print(f"Episode: {episode},\tScore: {score},\tEps: {agent.epsilon:4f}")

with open('models\\scores.txt', 'w') as f:
    f.write("\n".join(map(str, scores)))

import matplotlib.pyplot as plt

plt.plot(range(len(scores)), scores)
plt.show()
