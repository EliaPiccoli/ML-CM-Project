import numpy as np

def compute_gamma(x, gamma):
    return 1/(x.shape[1]*x.var()) if gamma == 'scale' else 1/x.shape[1] # scale - auto

def rbf(v1, v2, gamma='scale'):
    if isinstance(gamma, str):
        gamma = compute_gamma(v1, gamma)
    K = np.zeros((v1.shape[0], v2.shape[0]))
    for i in range(len(K)):
        for j in range(len(K[0])):
            K[i,j] = np.exp(-gamma * np.linalg.norm(v1[i]-v2[j])**2)
    return K, gamma

def linear(v1, v2):
    K = np.zeros((v1.shape[0], v2.shape[0]))
    for i in range(len(K)):
        for j in range(len(K[0])):
            K[i,j] = v1[i].dot(v2[j])
    return K, None

def poly(v1, v2, gamma='scale', deg=3, coef=0.0):
    if isinstance(gamma, str):
        gamma = compute_gamma(v1, gamma)
    K = np.zeros((v1.shape[0], v2.shape[0]))
    for i in range(len(K)):
        for j in range(len(K[0])):
            K[i,j] = (gamma * v1[i].dot(v2[j]) + coef) ** deg
    return K, gamma

def sigmoid(v1, v2, gamma='scale', coef=0.0):
    if isinstance(gamma, str):
        gamma = compute_gamma(v1, gamma)
    K = np.zeros((v1.shape[0], v2.shape[0]))
    for i in range(len(K)):
        for j in range(len(K[0])):
            K[i,j] = np.tanh(gamma*v1[i].dot(v2[j]) + coef)
    return K, gamma

def get_kernel(model):
    if model.kernel == 'linear':
        return linear(model.xs, model.xs)
    elif model.kernel == 'rbf':
        return rbf(model.xs, model.xs, model.gamma)
    elif model.kernel == 'poly':
        return poly(model.xs, model.xs, model.gamma, model.degree, model.coef)
    elif model.kernel == 'sigmoid':
        return sigmoid(model.xs, model.xs, model.gamma, model.coef)