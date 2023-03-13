import numpy as np
import matplotlib.pyplot as plt
import math

class Data():
    def __init__(self, avg, std):
        self.avg = avg
        self.std = std
        self.data = np.array([]).astype(int)
    def generate(self, length):
        data = np.rint(np.random.normal(size=length, loc=self.avg, scale=self.std)).astype(int)
        return np.where(data<0, 0, data)
    def getsample(self, dcount):
        if len(self.data) < dcount:
            self.data = np.concatenate((self.data, self.generate(dcount - len(self.data))))
            return self.data[dcount]
        return self.data[dcount]
    def getsample(self, startDay, length=1):
        if len(self.data) < startDay+length:
            self.data = np.concatenate((self.data, self.generate(startDay+length - len(self.data))))
        return self.data[startDay:startDay+length]
    



