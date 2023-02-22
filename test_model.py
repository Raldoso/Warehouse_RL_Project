from enviroment import WarehouseEnv, Store
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
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
    epsilon_min=    1.0,
    temperature=    5,#not important atm
    batch_size=     3,
    memory_size=    100,
    target_update_rate=50,
    policy_save_rate=20,
)

agent.load_model(r"C:\Users\ASUS\Downloads\ItWork\Projects\Udemy_PyML_Bootc\LIDL_ML_Procect\models\300_(-700.0)_warehouse_agent.pth")
state = env.reset()
for j in range(300):
    action = agent.choose_action(state,simulate=True)
    #action = max(0, sum([store.avg for store in env.stores]) - sum([sum(store.storage) for store in env.stores]))
    
    # print(state,action,end="\t")
    state, reward, done, error = env.step(action)
    # print(reward)
    if done:
        break

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1,constrained_layout=True)

recieved = env.stores[0].history[:,0]
storage = env.stores[0].history[:,1]
bought = env.stores[0].history[:,2]
overbuy = env.stores[0].history[:,3]
ordered = env.stores[0].history[:,4]
expired = env.stores[0].history[:,5]

actions = env.stores[0].history[:,0]#+env.stores[0].history[:,0]+env.stores[0].history[:,0]
# plt.hist(actions, bins=env.maxorder)

ax1.plot(recieved,"tab:green")
ax1.set_title("Recieved")
ax1.grid()

ax2.plot(storage,"tab:blue")
ax2.set_title("Sum of storage")
ax2.grid()

ax3.plot(bought,"tab:grey")
ax3.set_title("Daily bought amount")
ax3.grid()

ax4.plot(overbuy,"tab:orange")
ax4.set_title("Overbuy")
ax4.grid()

ax5.plot(ordered,"tab:purple")
ax5.set_title("Store status")
ax5.grid()

ax6.plot(expired,"tab:red")
ax6.set_title(" Daily expired items")
ax6.grid()

plt.show()