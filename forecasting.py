import numpy as np

class Forecast():
    def __init__(self, avg, std, pred_len):
        self.avg = avg
        self.std = std
        self.data = np.rint(np.random.normal(size=pred_len, loc=self.avg, scale=self.std))
        self.data = [+0 if num < 0 else num for num in self.data]

    def update(self):
        self.data = np.roll(self.data, -1)
        self.data[-1] =  max(np.rint(np.random.normal(size=1, loc=self.avg, scale=self.std)), 0) # type: ignore

    
if __name__ == '__main__':
    data = Forecast(13, 5, 5)
    for i in range(10):
        print(data.data)
        data.update()


