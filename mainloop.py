from enviroment import WarehouseEnv
from store import Store
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np

NUM_DAYS = 500
NUM_SIM_DAYS = 50
NUM_EPISODES = 1000
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
        max_age=6,
        min_items_percent=0.1))
env.addStore(
    Store(
        avg=13,
        std=5,
        max_age=6,
        min_items_percent=0.1))
env.addStore(
    Store(
        avg=20,
        std=5,
        max_age=6,
        min_items_percent=0.1))
env.setup_spaces()

sim_env = WarehouseEnv(
    max_age=6,
    n_days=NUM_SIM_DAYS
    )
sim_env.addStore(
    Store(
        avg=8,
        std=5,
        max_age=6,
        min_items_percent=0.1))
sim_env.addStore(
    Store(
        avg=13,
        std=5,
        max_age=6,
        min_items_percent=0.1))
sim_env.addStore(
    Store(
        avg=20,
        std=5,
        max_age=6,
        min_items_percent=0.1))
sim_env.setup_spaces()

"""
SETUP AGENT
"""
scores = [] #per episod
epsilones = [] #per episode
episode_losses = []
day_losses = []
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
    target_update_rate=100,# Steps between Target Copy
    policy_save_rate=20,#   Episodes between Saving Policy
)
"""
LOOP
"""
fig, ax = plt.subplots(figsize=(13, 5))
ax.grid(True)
ax.set_title("Cumulative Reward History")
ax.set_xlabel("Episode")
ax.set_ylabel("Score")
ax.tick_params(axis='y', labelcolor='tab:blue')
ax.yaxis.label.set_color('tab:blue')

ax1 = ax.twinx()
ax1.set_ylabel("Epsilon")
ax1.tick_params(axis='y', labelcolor='tab:orange')
ax1.yaxis.label.set_color('tab:orange')

ax2 = ax.twinx()
ax2.set_ylabel("Simulation Score")
ax2.tick_params(axis='y', labelcolor='tab:grey')
ax2.yaxis.label.set_color('tab:grey')
ax2.spines["right"].set_position(("axes", 1.1))

# agent.load_model(r"C:\Users\ASUS\Downloads\ItWork\Projects\Udemy_PyML_Bootc\LIDL_ML_Procect\models\360_(105493.0)_warehouse_agent.pth")
for episode in range(1,NUM_EPISODES+1):
    score = 0
    day_losses = []
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
            loss=agent.update()
            day_losses.append(loss)
        
        state = next_state
        step += 1

    if error:break
    
    if episode % 500 == 0:
        agent.epsilon=1
    
    #SIMULATION
    sim_score = 0
    sim_state = sim_env.reset()
    for day in range(NUM_SIM_DAYS):
        action, q_values = agent.choose_action(sim_state,simulate=True)
        sim_state, sim_reward, done, error = sim_env.step(action)
        sim_score += sim_reward

    #PRINTING AND LOGGING
    epsilones.append(agent.epsilon)
    scores.append(sim_score)
    episode_losses.append(np.mean(day_losses))
    if episode % agent.policy_save_rate == 0:
        agent.save_model(f"{episode}_({score})_warehouse_agent")
    
    print(f"Episode: {episode:>4.0f},\tAvg Rw: {sim_score//NUM_DAYS:>7.0f}\tEps: {agent.epsilon:.3f}\tAvg Loss: {np.mean(day_losses):.3f}")

    ax.plot(range(len(scores)),scores,color="tab:blue")
    ax1.plot(range(len(epsilones)),epsilones,color="tab:orange")
    ax2.plot(range(len(episode_losses)),episode_losses,color="tab:grey")
    plt.pause(0.001)

with open('models\\scores.txt', 'w') as f:
    f.write("\n".join(map(str, scores)))

plt.tight_layout()
plt.show()