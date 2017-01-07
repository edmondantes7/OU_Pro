# This will simualte random paths using an OU Process

# This file has been updated on 01/06/2017 to include geometric brownian motion and the file has been renamed on github. 

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

sp = pd.DataFrame(np.zeros((200,3)))
spm = pd.DataFrame(np.zeros((200,3)))
spou = pd.DataFrame(np.zeros((200,3)))
spsk = pd.DataFrame(np.zeros((200,3)))

ret = pd.DataFrame(np.zeros((200,3)))
retm = pd.DataFrame(np.zeros((200,3)))
retou = pd.DataFrame(np.zeros((200,3)))
retsk = pd.DataFrame(np.zeros((200,3)))

end = 100

sp.iloc[1,1] = 100
mu = 0
delta = 1
sigma = .05


# Generate GBM with Drift Process ----------------------------------------------------------------------------------------------
for i in range(2,100):
	sp.iloc[i,1] = sp.iloc[i-1,1] + mu*delta + sigma*math.sqrt(delta)*np.random.normal(0,1)
	
# Calculate GBM return vector
for i in range(2,100):
	ret.iloc[i,1] = (sp.iloc[i,1]/sp.iloc[i-1,1])-1

#-------------------------------------------------------------------------------------------------------------------------------
# Generate Geometric GBM with Drift Process (This is a work in progress - Still Incomplete)

#for i in range(2,100):
#	sp.iloc[i,1] = sp.iloc[i-1,1] + mu*delta + sigma*math.sqrt(delta)*np.random.normal(0,1)
	
# Calculate GBM return vector
# for i in range(2,100):
#	ret.iloc[i,1] = (sp.iloc[i,1]/sp.iloc[i-1,1])-1


# Generate synthetic momentum Process -------------------------------------------------------------------------------------

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

spsk.iloc[1,1] = 100

for i in range(2,100):
	spsk.iloc[i,1] = spsk.iloc[i-1,1] + mu*delta + sigma*math.sqrt(delta)*np.random.normal(0,1)
	
# Calculate GBM return vector
for i in range(2,100):
	retsk.iloc[i,1] = (spsk.iloc[i,1]/spsk.iloc[i-1,1])-1


#------Graphs and Chart Production

#1 Graph of the GBM process
fig = plt.figure(figsize=(16,12))
sp.iloc[1:(end-1),1].plot()
plt.title("GBM Simulation")
plt.savefig('graphs/GBM Figure 1')

#1 Distribution of GBM return chart
fig = plt.figure();
ret.iloc[1:(end-1),1].diff().hist(bins=50)
plt.title("GBM Distribution")
plt.savefig('graphs/GBM Dist 1')

#2 Graph of the Momentum process
fig = plt.figure(figsize=(16,12))
spm.iloc[20:(end-1),1].plot()
plt.title("SynMom Simulation")
plt.savefig('graphs/SynMom Figure 1')

#2 Distribution of Momentum return chart
fig = plt.figure();
retm.iloc[20:(end-1),1].diff().hist(bins=20)
plt.title("SynMom Distribution")
plt.savefig('graphs/SynMom Dist 1')

#3 Graph of the OU process
fig = plt.figure(figsize=(16,12))
spou.iloc[1:(end-1),1].plot()
plt.title("OU Simulation")
plt.savefig('graphs/OU Figure 1')

#3 Distribution of OU return chart
fig = plt.figure();
retou.iloc[1:(end-1),1].diff().hist(bins=20)
plt.title("OU Distribution")
plt.savefig('graphs/OU Dist 1')

#4 Graph of the Skew process
fig = plt.figure(figsize=(16,12))
spsk.iloc[1:(end-1),1].plot()
plt.title("Skew Simulation")
plt.savefig('graphs/Skew Figure 1')

#4 Distribution of Skew return chart
fig = plt.figure();
retsk.iloc[1:(end-1),1].diff().hist(bins=20)
plt.title("Skew Distribution")
plt.savefig('graphs/Skew Dist 1')#

