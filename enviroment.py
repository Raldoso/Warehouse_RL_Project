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
            self.maxorder += max(store.avg_range) * 3
            self.state_size += store.max_age + 2
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

            store.one_day(provided)
            
            reward -= store.expired * 100
            reward += 15 * (sum(store.storage) - store.min_items)
            reward -= sum(store.storage * (np.arange(len(store.storage))+1)) #the older the item the more -points it gets

            observation.extend(store.storage)
            observation.append(provided)
            observation.append(store.ordered_amount)
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
        
        self.avg = avg_range[0]
        self.std = std_range[0]
        self.overbuy = 0
        self.min_items = 0
        self.min_items_percent = min_items_percent
        self.recieved = 0

        # start with empty storage and order average for fillup
        self.ordered_amount = self.avg
        self.expired = 0
        self.storage = np.zeros(self.max_age)
        
        self.history = np.zeros((0,6)) #recieved,storage_before,bought ,overbuy, ordered, expired
        self.record = np.zeros(6)

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
        #self.avg = 0
        #self.std = 0
        self.overbuy = 0
        self.ordered_amount = 0
        self.storage[:] = 0
        self.recieved = 0
        self.expired = 0
        self.history = np.zeros((0,6))

    def update_storage(self):
        """
        Subtract demand from storage and return the supply-demand difference
        """

        # get daily demand amount from distribution
        bought_amount = self.get_sold_amount()
        self.record[2] = bought_amount

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
        self.expired = self.storage[-1]
        self.storage = np.roll(self.storage, 1)
        self.storage[0] = 0

        return bought_amount

    def one_day(self, received):
        """
        - simulate one day for the Store
        - recieving supply
        - update storage for the demand of the day
        - place new order
        """
        self.record[1] = np.sum(self.storage)
        self.storage[0] = received
        self.record[0] = received

        # process supply and demand
        self.overbuy = self.update_storage()
        # calculate order for next day
        self.ordered_amount = self.avg - sum(self.storage)
        
        self.ordered_amount = max(self.ordered_amount, 0)
        
        self.record[3] = self.overbuy
        self.record[4] = self.ordered_amount
        self.record[5] = self.expired
        self.record = np.reshape(self.record, (-1,6))
        self.history = np.append(self.history,self.record,axis=0)
        self.record = np.zeros(6)

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
    w.setup_spaces()
    # print(w.observation_space.sample())
    # print(w.action_space.sample())
    print(w.maxorder)
    print(w.state_size)
    import random
    #[w.step(7) for _ in range(100)]
    x = np.zeros((0,4))
    print(x)
    x = np.append(x,np.array([[1,2,3,4]]),axis=0)
    x = np.append(x,np.array([[1,2,3,4]]),axis=0)
    x = np.append(x,np.array([[1,2,3,4]]),axis=0)
    y = np.zeros(4)
    y = np.reshape(y,(-1,4))
    print(y)
    x = np.append(x,y,axis=0)
    print(x)
