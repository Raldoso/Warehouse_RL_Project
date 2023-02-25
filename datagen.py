import numpy as np
import matplotlib.pyplot as plt
import math

def plot_time_series(time, values, label):
    plt.figure(figsize=(10,6))
    plt.plot(time, values)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.title(label, fontsize=20)
    plt.grid(True)


time = np.arange(50)
values = np.where(time<10, time*3, (time-9)**2)

seasonal = []
for i in range(5):
    for j in range(50):
        seasonal.append(values[j])
time_seasonal = np.arange(250)


def sDataGenerator(length, max, weekly=True, monthly=True, trend=True):
    if(trend):
        data = np.arange(length)*(max*0.05/length)+max/2
    else:
        data = np.ones(length) * max/3
    if(weekly):  
        time = np.arange(7)
        values = ((time-2)**2)/20
        data += np.tile(values, math.ceil(length/7))[:length]
    if(monthly):
        time = np.arange(30)
        values = ((time-13)**2)/100
        data += np.tile(values, math.ceil(length/30))[:length]
    
    noise = (np.random.rand(length))/4+0.875
    data *= noise
    data = np.where(data<0, 0, data)
    data = np.where(data>max, max, data)
    return data


def dataGenerator(length, max, noiseStrength=0.2, resolution = 10):
    data = np.ones(length) * max/2
    gradients = np.random.rand(math.ceil(length/resolution)+1)*noiseStrength*2 + (1-noiseStrength)
    for i in range(length):
        p1 = i//resolution
        p2 = p1 + 1 
        d1 = i/resolution-p1
        d2 = d1-1
        a1 = gradients[p1] * d1
        a2 = gradients[p2] * d2

        amt = 6 * d1**5 - 15*d1**4 + 10*d1**3

        lerp = amt * (a2 - a1) + a1
        data[i] *= (1+lerp)
    return data


if __name__ == "__main__":
    data = dataGenerator(310, 13, noiseStrength=0.2, resolution=10)
    plt.plot(data)
    plt.grid(True)
    plt.show()



