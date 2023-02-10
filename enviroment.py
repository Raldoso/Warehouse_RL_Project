import numpy as np

import gymnasium as gym
from gymnasium import spaces
import math as m

class WarehouseEnv():

    def __init__(self, max_age=7, n_days=500):
        self.daycount = 0
        self.stores = []
        self.max_age = max_age
        self.n_days = n_days
        self.storage = np.zeros(self.max_age + 4)
        
        #! error for debugging
        self.error = False
        
        self.maxorder = 0
        self.state_size = 0

    def setup_spaces(self):
        """ 
        Initialize observation and action spaces
        to mach Store paramters.
        """
        # give upper bound for the Warehouse orders
        # calculate state size
        for store in self.stores:
            self.maxorder += max(store.avg_range) * 3
            self.state_size += store.max_age + 1
        self.state_size += self.max_age + 1
            # print(f"State size: {self.state_size}")
                
            # self.observation_space = spaces.Dict(
            #     {
            #         #store info
            #         # a matrix sized (2, numStores)
            #         # first row is the sum if items in the store
            #         #2nd row it the last recieved amount if items in the store

            #         "store_info":   spaces.Box(low=np.array([0, 0]), high=np.array([200, 100]), dtype=np.int32),

            #         # TODO calculate store info max
    
            #         #warehouse info
            #         #contains 2 values, sum of storage, last ordered amount

            #         "warehouse_info": spaces.Box(low=0, high=self.maxorder, shape=(self.max_age+4,), dtype=np.int32)
            #     })
            # #maximum amount to order
            # self.action_space = spaces.Discrete(self.maxorder+1)
        pass

    def _get_obs(self):
        store = np.zeros(2, len(self.stores))
        for i, store in enumerate(self.stores):
            store[0][i] = store.storage
            store[1][i] = store.recieved
        wh_info = self.storage

        return {"store_info": store, "warehouse_info": wh_info}

    def step(self, action):
        """ 
        - Distribute by category favouring Stores at the beginning of queue.
        - A category is a group of items with the same age.
        - Because one Store can get items from multiple categories (it recieves a list)
        we build up the list with the supplied items for every store before
        it can be passed to the Stores.
        """
        reward = 0

        # * RECIEVEING AND DISTRIBUTING
        order_builder = np.zeros(shape=(len(self.stores), self.max_age))
        order_sum = sum([x.ordered_amount for x in self.stores])  # get global order for warehouse
        index = 0

        """
        Creating array of items from categories until 
        Store got what they ordered 
        or Warehouse goes empty 
        """
        while (order_sum > 0 and sum(self.storage) > 0):
            if self.storage[index + 3] >= order_sum:
                for i, store in enumerate(self.stores):
                    order_builder[i][index] = store.ordered_amount

                    self.storage[index + 3] -= store.ordered_amount
                break
            else:
                for i, store in enumerate(self.stores):
                    try:
                        provided = m.floor(self.storage[index + 3] * (store.ordered_amount / order_sum))
                    except Exception as e:
                        print(e)
                        print()    
                        print(self.storage)
                        print(order_sum)
                        print(store.ordered_amount)
                        
                        self.error = True
                        break
                    order_builder[i][index] = provided
                    store.ordered_amount -= provided
                    order_sum -= provided

                # this iteration ensures to give out
                # the whole category and the usage of floor() 
                # can result in remaining items
                self.storage[index + 3] = 0
                
                #only go to next category if not enough storage
                index += 1
                if(index >= self.max_age):
                    break
        
        # give out orders to the Stores, calculate rewards
        expired = 0
        observation = []
        # print(order_builder)
        for i, store in enumerate(self.stores):
            store.one_day(order_builder[i])
            
            reward -= store.expired * 100
            reward += 15 * sum(store.storage) - store.min_items
            reward -= sum(store.storage * (np.arange(len(store.storage))+1)) #the older the item the more -points it gets
            
            observation.extend(store.storage)
            observation.append(sum(order_builder[i]))
            expired += store.expired

        observation.extend(self.storage[4:])
        observation.append(expired)
        # print(observation,self.storage)
        
        # * PREPARATIONS
        reward -= 100 * self.storage[-1]
        self.storage = np.roll(self.storage, 1)
        self.daycount += 1
        self.storage[0] = action
        #(observation, reward, done, info)
        #observation = self._get_obs
        return observation, reward, self.daycount == self.n_days, self.error

    def addStore(self, store):
        """
        Connect new store to the Warehouse
        """
        #self.storage += store.avg
        self.stores.append(store)

    def reset(self):
        self.daycount = 0
        self.storage[:] = 0
        for store in self.stores:
            store.reset()
        return [0]*self.state_size

class Store():
    def __init__(self, avg_range, std_range, max_age=6,min_items_percent=0.2):

        self.avg_range = avg_range
        self.std_range = std_range
        self.max_age = max_age
        
        self.avg = 0
        self.std = 0
        self.overbuy = 0
        self.min_items = 0
        self.min_items_percent = min_items_percent
        self.recieved = 0

        # start with empty storage and order average for fillup
        self.ordered_amount = self.avg
        self.expired = 0
        self.storage = np.zeros(self.max_age)

    def get_sold_amount(self):
        """
        Create variance in daily distribution by picking random for the average and deviation.
        Recalculate minimum storage to stay at the end.
        The enforcement of min_items happens within the reward system
        """

        # update distribution for current day
        self.avg = np.random.choice(self.avg_range)
        self.std = np.random.choice(self.std_range)
        self.min_items = round(self.avg * self.min_items_percent)

        # return bought amount for the day
        return max(round(np.random.normal(self.avg, self.std, 1)[0], 0), 0)

    def reset(self):
        self.avg = 0
        self.std = 0
        self.overbuy = 0
        self.ordered_amount = 0
        self.storage[:] = 0
        self.recieved = 0
        self.expired = 0

    def update_storage(self):
        """
        Subtract demand from storage and return the supply-demand difference
        """

        # get daily demand amount from distribution
        bought_amount = self.get_sold_amount()

        # rolling subtraction of demand from storage
        for index in range(self.max_age):
            bought_amount -= self.storage[index]
            if (bought_amount >= 0):
                self.storage[index] = 0
            else:
                self.storage[index] = (-1) * bought_amount
                bought_amount = 0
                break
        # get too old items
        # shift storage by one day
        self.expired = self.storage[self.max_age - 1]
        self.storage = np.roll(self.storage, 1)
        self.storage[0] = 0

        return bought_amount

    def one_day(self, received: np.ndarray):
        """
        - simulate one day for the Store
        - recieving supply
        - update storage for the demand of the day
        - place new order
        """
        #reshaping recieved array, and adding it to storage

        reshaped = np.zeros(self.max_age)
        reshaped[:len(received)] = received
        self.recieved = sum(received)
        self.storage += reshaped

        # process supply and demand
        self.overbuy = self.update_storage()
        # calculate order for next day
        self.ordered_amount = self.avg - sum(self.storage)
        
        self.ordered_amount = max(self.ordered_amount, 0)

if __name__ == '__main__':
    #testing
    w = WarehouseEnv(
        max_age=6,
        n_days=500
    )
    w.addStore(Store(
        avg_range=[8],
        std_range=[5],
        max_age=6)) 
    # w.addStore(Store(
    #     avg_range=[13],
    #     std_range=[5],
    #     max_age=6))
    # w.addStore(Store(
    #     avg_range=[20],
    #     std_range=[5],
    #     max_age=6))
    w.setup_spaces()
    # print(w.observation_space.sample())
    # print(w.action_space.sample())
    print(w.maxorder)
    import random
    [w.step(7) for _ in range(100)]
