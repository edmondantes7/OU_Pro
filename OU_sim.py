# This will simualte random paths using an OU Process

# This is a new file that has been created on 5/7/2016

import numpy as np
import pandas as pd 
from math import exp
from math import sqrt
randn = np.random.randn


head=list(range(0,100))
rows=list(range(0,100))



path1 = pd.DataFrame(np.zeros((100,100)),index=head, columns=rows)


path1[0,1] = 1
delta1 = .25
lambda1 = .4
mu1 = .03
sig1 = .30


for x in range(0,90):
	path1.loc[x+1,0] = path1.loc[x,0]*exp(-lambda1*delta1) + mu1*(1-exp(-lambda1*delta1)) + (sig1*sqrt((1-exp(-2*lambda1*delta1))/(2*lambda1)))*randn()

path1.to_csv('path1_test.csv')


# Make change to test on github.
