import numpy as np
import matplotlib.pyplot as plt
import math

def autocovariance(k, H):
    return 0.5 * (abs(k - 1) ** (2 * H) - 2 * abs(k) ** (2 * H) + abs(k + 1) ** (2 * H))

def cholesky_method(N, H):
    gn_list = np.random.normal(0.0, 1.0, N)
    G = np.zeros([N, N])
    for i in range(N):
        for j in range(i + 1):
            G[i, j] = autocovariance(i - j, H)

    C = np.linalg.cholesky(G)

    fgn_list = np.dot(C, np.array(gn_list).transpose())
    fgn_list = np.squeeze(fgn_list)
    return fgn_list

def fgn(fgn_l, H):
    scale = (1.0 / len(fgn_l)) ** H
    return fgn_l * scale

def fbm(Chol_list):
    return np.insert(fgn(Chol_list, H).cumsum(), [0], 0)

N = 1000
T = 1
H_values = [0.1, 0.3, 0.5, 0.7]
t = np.linspace(0, T, N + 1)

fig, axs = plt.subplots(4, 1, figsize=(10, 15))

for i, H in enumerate(H_values):
    Chol_list = cholesky_method(N, H)
    fbm_list = fbm(Chol_list)
    axs[i].plot(t, fbm_list)
    axs[i].axis('off')

plt.tight_layout()
plt.show()
