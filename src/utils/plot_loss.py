import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys


data = np.loadtxt(sys.argv[1])
data_1 = np.loadtxt(sys.argv[2])
data_2 = np.loadtxt(sys.argv[3])
data_3 = np.loadtxt(sys.argv[4])

fig,ax = plt.subplots()
plt.plot(data[:100,1],label='001000')
ax.plot(data_1[:100,1],label='001010')
ax.plot(data_2[:100,1],label='111010')
ax.plot(data_3[:100,1],label='111111')
#plt.plot(100*data[:1000,1],label='rotation loss * 100 w1')
'''
data = np.loadtxt(sys.argv[2])

plt.plot(data[:1000,0],label='translation loss w10')
plt.plot(100*data[:1000,1],label='rotation loss * 100 w10')

data = np.loadtxt(sys.argv[3])

plt.plot(data[:1000,0],label='translation loss w100')
plt.plot(100*data[:1000,1],label='rotation loss * 100 w100')
'''
ax.set_xlabel('batch number')
ax.set_ylabel('loss')
ax.set_yscale('logit')
#ax.set_yticks([0.001,0.01,0.1,1])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_title('Random training loss visualization of different motion axis first 100 epoch')
ax.legend()
plt.show()
