
from gymenv import WareHouseEnv
from store import Store
import gymnasium as gym

stores = [
    Store(avg = 14, std = 2, max_age = 5, min_items_percent = .1),
    Store(avg = 3, std = 2, max_age = 1, min_items_percent = .1),
    Store(avg = 5, std = 2, max_age = 7, min_items_percent = .1)]

env = WareHouseEnv(stores)
observation, info = env.reset()
for _ in range(1000):
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
