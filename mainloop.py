from enviroment import WarehouseEnv, Store
from agent import Agent
import numpy as np

NUM_EPISODES = 10000

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
        max_age=5)) 
env.addStore(
    Store(
        avg_range=[13],
        std_range=[5],
        max_age=5))
env.addStore(
    Store(
        avg_range=[20],
        std_range=[5],
        max_age=5))
env.setup_spaces()

""" 
SETUP AGENT
"""
agent = Agent(...)

""" 
HISTORY
"""
scores = []


for episode in range(NUM_EPISODES):
    score = 0
    done = False
    state = env.reset()

    while not done:
        """ 
        1. Choose action
        2. Pass action to enviroment
        3. Recieve new state and reward from enviroment
        4. Save observation and reward
        5. Update Agent network
        """
        
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        #use the cumulated reward as an overall score for this episode
        score += reward
        
        agent.memory.add_transition(state, action, reward, next_state)
        
        # We pass batches of observations to the agent
        # so we need at least one batch amount of observations
        if len(agent.memory) > agent.batch_size:
            agent.update()
        
        state = next_state
        
    scores.append(score)
    if episode % 100 == 0:
        # Print out performance after every 100 episodes
        print(f"Episode: {episode}, Score: {score} , Avg Score: {np.mean(scores[-100:])}")


# Save the model at the and of training to reuse later
agent.save_model()

