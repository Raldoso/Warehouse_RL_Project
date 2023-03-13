import numpy as np
import math as m


class WarehouseEnv():

    def __init__(self, max_age=7, n_days=500):
        self.daycount = 0
        self.stores = []
        self.max_age = max_age
        self.n_days = n_days
        self.storage = np.zeros(5)
        
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
            self.maxorder += store.avg * 3
            self.state_size += store.max_age + 4
        self.state_size += 1
        

    def step(self, action):
        reward = 0

        # * RECIEVEING AND DISTRIBUTING
        order_sum = sum([x.ordered_amount for x in self.stores])  # get global order for warehouse
        avg_sum = sum([x.avg for x in self.stores])

        # give out orders to the Stores, calculate rewards
        expired = 0
        observation = []
        for i, store in enumerate(self.stores):
            
            if order_sum != 0:
                provided = m.floor(self.storage[4]*(store.ordered_amount/order_sum))
            else:
                provided = m.floor(self.storage[4]*(store.avg/avg_sum))

            store.one_day(provided, self.daycount)
            
            reward -= store.expired * 100
            reward += 15 * (sum(store.storage) - store.min_items)
            reward -= sum(store.storage * (np.arange(len(store.storage))+1)) #the older the item the more -points it gets

            observation.extend(store.storage)
            observation.extend(store.data.getsample(self.daycount, length=4))
            expired += store.expired

        observation.append(expired)
        
        # * PREPARATIONS
        # reward -= 30 * self.storage[-1]
        self.storage = np.roll(self.storage, 1)
        self.daycount += 1
        self.storage[0] = action
        return observation, reward, self.daycount == self.n_days, self.error

    def addStore(self, store):
        """
        Connect new store to the Warehouse
        """
        #self.storage += store.avg
        self.stores.append(store)
        store.n_days = self.n_days

    def reset(self):
        self.daycount = 0
        self.storage[:] = 0
        for store in self.stores:
            store.reset()
        return [0]*self.state_size
