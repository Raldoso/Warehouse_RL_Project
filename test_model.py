from enviroment import WarehouseEnv, Store
import matplotlib.pyplot as plt
from agent import Agent
import numpy as np

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
RELOAD MODEL AND TEST AGENT
"""

NUM_EPISODES = 300
scores = []
agent = Agent(
    state_size=env.state_size,
    action_size=env.maxorder,
    learn_rate=     0.001,
    gamma=          0.99,
    epsilon_decay=  0.996,
    epsilon_min=    0.01,
    temperature=    5,#not important atm
    batch_size=     3,
    memory_size=    100,
    target_update_rate=50,
)

agent.load_model("models\\warehouse_agent.pth")
state = env.reset()
for j in range(300):
    action = agent.choose_action(state)
    state, reward, done, error = env.step(action)
    if done:
        break

overbuy = env.stores[0].history[:,0]
#print(env.stores[0].history)
plt.plot(range(len(overbuy)), overbuy)
plt.show()