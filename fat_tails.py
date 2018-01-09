# This is a script to play with Monte Carlos of different distributions
# R Dewey 7/7/2017

import sys
import cmath
import numpy as np
import pandas as pd
import csv
import scipy as sp
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm

from scipy.optimize import minimize
from numpy import matrix
from numpy.linalg import inv


# Cauchy Distribution

s = np.random.standard_cauchy(1000000)
s = s[(s>-25) & (s<25)]

plt.hist(s, bins=100)
plt.show()


# Pareto Distribution

a, m = 3., 2.  # shape and mode
q = (np.random.pareto(a, 1000) + 1) * m

count, bins, _ = plt.hist(q, 100, normed=True)
fit = a*m**a / bins**(a+1)
plt.plot(bins, max(count)*fit/max(fit), linewidth=2, color='r')
plt.show()


