# Code for simualting various types of stochastics porcesses

import datetime
import sys
import csv
import matplotlib.pyplot as plt 
#from matplotlib.backends.backend_pdf import Pdfpages

import random
import pandas as pd
import scipy as sp
import numpy as np
import math 
from math import exp
from math import sqrt


x = random.random()
print(x)

sp = pd.DataFrame(np.zeros((200,3)))   #standard brownian motion
spg = pd.DataFrame(np.zeros((200,3)))  #geometric brownian motion
spm = pd.DataFrame(np.zeros((200,3)))  # brownian with momentum
spou = pd.DataFrame(np.zeros((200,3))) # brownian with mean reversion
spsk = pd.DataFrame(np.zeros((200,3))) # brownian with a skewed distribution
spj = pd.DataFrame(np.zeros((200,3)))  # brownian with jumps

ret = pd.DataFrame(np.zeros((200,3)))
retg = pd.DataFrame(np.zeros((200,3)))
retm = pd.DataFrame(np.zeros((200,3)))
retou = pd.DataFrame(np.zeros((200,3)))
retsk = pd.DataFrame(np.zeros((200,3)))
retj = pd.DataFrame(np.zeros((200,3)))

end = 100

sp.iloc[1,1] = 100
mu = 0
delta = 1
sigma = .05

# Generate Brownian Motion process vector -----------------------------------------------------------------------------------------

# set mu term to determine the drift of the process
mu = 0

for i in range(2,200):
	sp.iloc[i,1] = sp.iloc[i-1,1] * (1+mu*delta + sigma*math.sqrt(delta)*np.random.normal(0,1))
print(sp)
	#sp.iloc[i,1] = sp.iloc[i-1,1] + mu*delta + sigma*math.sqrt(delta)*np.random.normal(0,1)

# Calculate BM return vector
for i in range(2,100):
	ret.iloc[i,1] = (sp.iloc[i,1]/sp.iloc[i-1,1])-1



# Generate Goemetric Brownian Motion process vector -----------------------------------------------------------------------------------------

# set mu term to determine the drift of the process
mu_g = 0

for i in range(2,200):
	spg.iloc[i,0] = spg.iloc[i-1,0] * math.exp((mu_g-0.5*sigma**2)*delta + sigma*math.sqrt(delta)*np.random.normal(0,1))
print(spg)

# Calculate GBM return vector
for i in range(2,100):
	retg.iloc[i,1] = (spg.iloc[i,1]/spg.iloc[i-1,1])-1

# Generate synthetic momentum process vector --------------------------------------------------------------------------

spm.iloc[20,1] = 100
lamda = .70
n = 20
sum1 = 0


for i in range(21,100):
	for k in range(i-n,i):
		sum1 = sum1 + sp.iloc[k,1]
	spm.iloc[i,1] = spm.iloc[i-1,1] + mu*delta + sigma*math.sqrt(delta)*np.random.normal(0,1) + delta*lamda*(sum1/n)*ret.iloc[i-1,1]

#Calculate synthetic momentum returns
for i in range(21,100):
	retm.iloc[i,1] = (spm.iloc[i,1]/spm.iloc[i-1,1])-1


# Generate an OU Process vector ----------------------------------------------------------------------------------------

spou.iloc[1,1] = 100
lambda1 = .05
delta1 = 1
sig1 = .05
mu1 = 100

for x in range(2,100):
	spou.iloc[x,1] = spou.iloc[x-1,1]*exp(-lambda1*delta1) + mu1*(1-math.exp(-lambda1*delta1)) + (sig1*sqrt((1-math.exp(-2*lambda1*delta1))/(2*lambda1)))*np.random.randn()

# calculate OU returns
for i in range(21,100):
	retou.iloc[i,1] = (spou.iloc[i,1]/spou.iloc[i-1,1])-1


# Generate a Skewed Distribution ---------------------------------------------------------------------------------------

# On the to-do list. This is placeholder code

# spsk.iloc[1,1] = 100

# for i in range(2,100):
# 	spsk.iloc[i,1] = spsk.iloc[i-1,1] + mu*delta + sigma*math.sqrt(delta)*np.random.normal(0,1)
	
# # Calculate GBM return vector
# for i in range(2,100):
# 	retsk.iloc[i,1] = (spsk.iloc[i,1]/spsk.iloc[i-1,1])-1


# Generate a Jump Diffusion process vector -----------------------------------------------------------------------------------------

jumps = pd.DataFrame(np.zeros((10000,2)))

def nextTime(rateParameter):
    #return -math.log(1.0 - random.random()) / rateParameter
    return random.expovariate(1/40.0)

for i in range(1,1000):
	jumps.iloc[i,1] = nextTime(1/40.0)
	jumps.iloc[i,1] = round(jumps.iloc[i,1],0)


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


# Calculate Jump return vector
for i in range(2,end):
	retj.iloc[i,1] = (spj.iloc[i,1]/spj.iloc[i-1,1])-1







#------Graphs and Chart Production

#1 Graph of the BM process
fig = plt.figure(figsize=(16,12))
sp.iloc[1:(end-1),1].plot()
plt.title("BM Simulation")
plt.savefig('graphs/BM Figure 1')

#1 Distribution of BM return chart
fig = plt.figure();
ret.iloc[1:(end-1),1].diff().hist(bins=50)
plt.title("BM Distribution")
plt.savefig('graphs/BM Dist 1')

#2 Graph of the GBM process
fig = plt.figure(figsize=(16,12))
spg.iloc[1:(end-1),1].plot()
plt.title("GBM Simulation")
plt.savefig('graphs/GBM Figure 1')

#2 Distribution of GBM return chart
fig = plt.figure();
retg.iloc[1:(end-1),1].diff().hist(bins=50)
plt.title("GBM Distribution")
plt.savefig('graphs/GBM Dist 1')

#3 Graph of the Momentum process
fig = plt.figure(figsize=(16,12))
spm.iloc[20:(end-1),1].plot()
plt.title("SynMom Simulation")
plt.savefig('graphs/SynMom Figure 1')

#3 Distribution of Momentum return chart
fig = plt.figure();
retm.iloc[20:(end-1),1].diff().hist(bins=20)
plt.title("SynMom Distribution")
plt.savefig('graphs/SynMom Dist 1')

#4 Graph of the OU process
fig = plt.figure(figsize=(16,12))
spou.iloc[1:(end-1),1].plot()
plt.title("OU Simulation")
plt.savefig('graphs/OU Figure 1')

#4 Distribution of OU return chart
fig = plt.figure();
retou.iloc[1:(end-1),1].diff().hist(bins=20)
plt.title("OU Distribution")
plt.savefig('graphs/OU Dist 1')

#5 Graph of the Skew process
fig = plt.figure(figsize=(16,12))
spsk.iloc[1:(end-1),1].plot()
plt.title("Skew Simulation")
plt.savefig('graphs/Skew Figure 1')

#5 Distribution of Skew return chart
fig = plt.figure();
retsk.iloc[1:(end-1),1].diff().hist(bins=20)
plt.title("Skew Distribution")
plt.savefig('graphs/Skew Dist 1')

#6 Graph of the Jump process
fig = plt.figure(figsize=(16,12))
spj.iloc[1:(end-1),1].plot()
plt.title("Jump Simulation")
plt.savefig('graphs/Jump Figure 1')

#6 Distribution of Jump return chart
fig = plt.figure();
retj.iloc[1:(end-1),1].diff().hist(bins=20)
plt.title("Jump Distribution")
plt.savefig('graphs/Jump Dist 1')
