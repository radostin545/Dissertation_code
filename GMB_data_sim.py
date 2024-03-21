import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import anderson

r = 0.1
sigma = 0.2
T = 1
step = 1000
S0 = 100
n = 1000 # number of sims
dt = T/step

St = np.exp(
    (r - sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size = (n,step))
)

St = np.hstack([np.ones((n,1)), St])
St = S0 * St.cumprod(axis = 1)
Xt = np.log(St) # log prices

time = np.linspace(0, T, step+1)

tt = np.full(shape=(n,step+1), fill_value=time)

Xt = np.vstack((tt[0], Xt))

np.savetxt("training_data_GBM.csv", Xt, fmt='%.5f', delimiter=",")
