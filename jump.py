# This is a script for generating Poisson Processes
# And by extension, jump diffusion processes.

import datetime
import sys
import csv
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import pandas as pd
import scipy as sp
import numpy as np
from math import exp
from math import sqrt
import math 
import random


jumps = pd.DataFrame(np.zeros((10000,2)))

def nextTime(rateParameter):
    #return -math.log(1.0 - random.random()) / rateParameter
    return random.expovariate(1/40.0)

for i in range(1,1000):
	jumps.iloc[i,1] = nextTime(1/40.0)
	jumps.iloc[i,1] = round(jumps.iloc[i,1],0)

#y = data.iloc[:,1].mean()
#print(y)

# Generate a Jump Diffusion Process

spj = pd.DataFrame(np.zeros((1000,3)))
retj = pd.DataFrame(np.zeros((1000,3)))

end = 1000
spj.iloc[1,1] = 100
mu = .06
delta = .0039 # This is simply 1 day in a year or 1/252
#sigma = .062  # The measure of 1 day of volatility. So if vol = 15% this is 15% * SQRT(1/252)
sigma = .15  # The measure of annual volatility
lamda = 20


counter = jumps.iloc[1,1]
k = 2

for i in range(2,end):
	if counter == i:
		jump = .05
		counter = jumps.iloc[k,1] + counter
		print(counter)
		k = k + 1
	else:
		jump = 0
	spj.iloc[i,1] = spj.iloc[i-1,1] * (1+mu*delta + sigma*math.sqrt(delta)*np.random.normal(0,1)) + jump*np.random.normal(0,1)

jumps.to_csv("Jump_Vector.csv")


# Calculate GBM return vector
for i in range(2,end):
	retj.iloc[i,1] = (spj.iloc[i,1]/spj.iloc[i-1,1])-1

# auto1 = sm.tsa.stattools.acf(ret.iloc[1:250,1], nlags=10)
# print(auto1)


# Graph of the GBM process
fig = plt.figure(figsize=(16,12))
sp.iloc[1:(end-1),1].plot()
plt.title("Jump Simulation")
plt.savefig('graphs/Jump Figure 1')

#1 Distribution of GBM return chart
fig = plt.figure();
ret.iloc[1:(end-1),1].diff().hist(bins=50)
plt.title("Jump Distribution")
plt.savefig('graphs/Jump Dist 1')

