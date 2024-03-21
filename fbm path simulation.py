import numpy as np
import matplotlib.pyplot as plt
import math

T = 1
H = 0.2
N = 1000
t = np.linspace(0, T, N + 1)
sigma0 = 0.2
xi = 1
alpha = 0.2
rho = 0.5
number_of_paths = 10
r = 0.05
K = 110

X_0 = math.log(100)


V = np.zeros((0,1000))

def fbm(Chol_list):
    return np.insert(fgn(Chol_list).cumsum(), [0], 0)

def fgn(fgn_l):
    scale = (1.0 * T / N) ** H
    return fgn_l * scale

def autocovariance(k):
    return 0.5 * (abs(k - 1) ** (2 * H) - 2 * abs(k) ** (2 * H) + abs(k + 1) ** (2 * H))

def cholesky_method():
    gn_list = np.random.normal(0.0, 1.0, N)
    G = np.zeros([N, N])
    for i in range(N):
        for j in range(i + 1):
            G[i, j] = autocovariance(i - j)

    C = np.linalg.cholesky(G)

    fgn_list = np.dot(C, np.array(gn_list).transpose())
    fgn_list = np.squeeze(fgn_list)
    return fgn_list, gn_list

def single_fbm_path():
    Chol_list, gn_list = cholesky_method()
    fbm_list = fgn(Chol_list)
    final = fbm(fbm_list)
    # plt.plot(t, final)
    return final, gn_list


fBm_Paths = np.zeros((0,1001))
sigma_t = np.zeros((0,1000))

def indep_Wiener_process():
    W_tilde = np.zeros((10,1000))
    for i in range(number_of_paths):
        rand_list = np.random.normal(0.0, 1.0, N)
        W_tilde[i] = rand_list * math.sqrt(1/1000)
    return W_tilde

def sigma(GV_process,t):
    t = t[1:]
    GV_process = np.squeeze(GV_process)
    sigma_t=[]
    for i in range(len(t)):
        ans = sigma0*math.exp(xi*GV_process[i]-0.5*alpha*xi**2*t[i]**(2*H))
        sigma_t.append(ans)
    return sigma_t

def call_option(S):
    call_payoff =[]
    for i in S[:,-1]:
        call_payoff.append(np.exp(-r * T)*np.maximum(i - 110, 0))
    return call_payoff, np.mean(call_payoff)

for i in range(number_of_paths):
    current_path, gn_list = single_fbm_path()
    current_path = np.array(current_path)
    current_path = current_path[np.newaxis]

    current_sigma = sigma(gn_list,t)
    current_sigma = np.array(current_sigma)
    current_sigma = current_sigma[np.newaxis]

    gn_list = np.array(gn_list)
    gn_list = gn_list[np.newaxis]
    V = np.r_[V,gn_list]

    # Simulating the P sample Volterra paths 
    fBm_Paths = np.r_[fBm_Paths,current_path]

    # Simulating the P sample volatility paths
    sigma_t = np.r_[sigma_t,current_sigma]


# W and W_tilde
driving_W = math.sqrt(1/1000)*V
indep_W = indep_Wiener_process()


Z = np.zeros((0,1000))
for j in range(number_of_paths):
    Z_j = rho*driving_W[j]+math.sqrt(1-rho**2)*indep_W[j]
    Z_j = Z_j[np.newaxis]
    Z = np.r_[Z, Z_j]


delta_Z  = np.zeros((10,999))
for j in range(number_of_paths):
    for i in range(999):
        delta_Z[j,i] = Z[j,i+1]-Z[j,i]

delta_X  = np.zeros((10,1000))
for j in range(number_of_paths):
    for i in range(1,N-2):
        delta_X[j,i+1] = (r - 0.5*sigma_t[j,i])*1/1000 + math.sqrt(sigma_t[j,i])*delta_Z[j,i+1]

cumulative = np.cumsum(delta_X, axis = 1)
X = X_0 + cumulative
S = np.exp(X)

call_payoff, call_average = call_option(S)
print(call_payoff, call_average)

fig, axs = plt.subplots(2,1,figsize=(12, 6))

for row in S:
    axs[0].plot(row)
axs[0].set_title('Simulated Call Option Paths')
S_averaged = np.mean(S, axis=0)
for i in S_averaged:
    axs[1].plot(S_averaged)
axs[1].set_title('Average of Simulated Call Option Paths')

plt.show()