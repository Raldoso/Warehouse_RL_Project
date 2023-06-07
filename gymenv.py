import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete
from store import Store
import math


class WareHouseEnv(gym.Env):
    def __init__(self, stores, pred_len = 4, n_days = 500):
        self.n_days = n_days
        self.pred_len = pred_len
        self.stores = stores
        self.storage = np.zeros(5)
        self.score = 0
        self.numStores = len(stores)
        self.max_age = max(stores, key=lambda store: store.max_age).max_age
        predictions = np.ones([self.numStores, pred_len])
        items = np.ones([self.numStores, self.max_age])
        for i, store in enumerate(stores):
            items[i][:store.max_age] = store.forecast.avg * 3
            predictions[i][:] = store.forecast.avg * 3
        self.observation_space = spaces.Dict(
            {
                "items": spaces.MultiDiscrete(items),
                "predictions": spaces.MultiDiscrete(predictions)
            }
        )
        """
        The resulting obseration space will be a dictionary of two arrays

        the first one conatining information about the store states
            the resulting observation space will be [num of stores] x [max age over all stores],
            and each value will be the maximum possible amount of that cell (now its 3 x avg)

        the second one about the predictions [num of stores] x [predictions length]
        i couldnt figure out a way to disable a cell in multidiscrete space
        so for now the max is 1, even if its not possible to have a value in that cell
        """
        self.max_action= sum(store.forecast.avg for store in stores) * 3
        self.action_space = spaces.Discrete(self.max_action * 3) #? im not sure, if this is a bit too high
    def _getobs(self):
        store_data = np.zeros([self.numStores, self.max_age])
        predictions_data = np.zeros([self.numStores, self.pred_len])
        for i, store in enumerate(self.stores):
            store_data[i][:store.max_age] = store.storage
            predictions_data[i][:] = store.forecast.data
        """
        formatting the data in the stores to be the same as in the observation space
        """
        return{
            "items": store_data,
            "predictions":predictions_data
        }
    def _getinfo(self):
        return self.score
    def reset(self):
        self.daycount = 0
        self.storage[:] = 0
        for store in self.stores:
            store.reset()
        observation = self._getobs()
        info = self._getinfo()
        return observation, info
    def step(self, action):
        reward = 0

        # * RECIEVEING AND DISTRIBUTING
        order_sum = sum([x.ordered_amount for x in self.stores])  # get global order for warehouse
        avg_sum = sum([x.forecast.avg for x in self.stores])
        
        for store in self.stores:
            
            if order_sum != 0:
                provided = math.floor(self.storage[4]*(store.ordered_amount/order_sum))
            else:
                provided = math.floor(self.storage[4]*(store.forecast.avg/avg_sum))

            store.one_day(provided)
            
            reward -= store.expired * 50
            reward += 15 * (sum(store.storage) - store.min_items)
            reward -= sum(store.storage * (np.arange(len(store.storage))+1))
            # type: ignore #the older the item the more -points it gets

        self.storage = np.roll(self.storage, 1)
        self.daycount += 1
        self.storage[0] = action
        observation = self._getobs()
        terminated = self.daycount == self.n_days
        self.score += reward
        info = self._getinfo()
        return observation, reward, terminated, False, info
    def close(self):
        return

