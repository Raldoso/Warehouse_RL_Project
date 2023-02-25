import numpy as np
from datagen import dataGenerator, sDataGenerator

class Store():
    def __init__(self, avg_range, std_range, max_age=6,min_items_percent=0.2,):

        self.avg_range = avg_range
        self.std_range = std_range
        self.max_age = max_age
        
        self.avg = avg_range[0]
        self.std = std_range[0]
        self.overbuy = 0
        self.min_items = 0
        self.min_items_percent = min_items_percent
        self.recieved = 0
        self.n_days = 0

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
        #self.pred = dataGenerator(self.n_days+10, max(self.avg_range) + max(self.std_range), noiseStrength=0.2, resolution=10)
        self.pred = sDataGenerator(self.n_days+10, max(self.avg_range) + max(self.std_range), weekly=False, monthly=False, trend=False)

    def update_storage(self, daycount):
        """
        Subtract demand from storage and return the supply-demand difference
        """

        # get daily demand amount from distribution
        bought_amount = self.pred[daycount]
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

    def one_day(self, received, daycount):
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
        self.overbuy = self.update_storage(daycount)
        # calculate order for next day
        self.ordered_amount = self.avg - sum(self.storage)
        
        self.ordered_amount = max(self.ordered_amount, 0)
        
        self.record[3] = self.overbuy
        self.record[4] = self.ordered_amount
        self.record[5] = self.expired
        self.record = np.reshape(self.record, (-1,6))
        self.history = np.append(self.history,self.record,axis=0)
        self.record = np.zeros(6)