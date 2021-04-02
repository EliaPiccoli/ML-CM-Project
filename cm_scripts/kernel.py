import numpy as np
def rbf(x, gamma):
    K = np.zeros((x.size, x.size))
    for i in range(len(K)):
        for j in range(len(K[0])):
            K[i,j] = np.exp(-np.linalg.norm(x[i]-x[j])**2/ (2*(gamma**2)))
    return K