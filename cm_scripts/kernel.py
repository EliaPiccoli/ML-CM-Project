import numpy as np

def compute_gamma(x, gamma):
    return 1/(x.shape[1]*x.var()) if gamma == 'scale' else 1/x.shape[1]

def rbf(x, gamma='scale', orig_x=None):
    if isinstance(gamma, str):
        if orig_x is not None:
            gamma = compute_gamma(orig_x, gamma)
        else:
            gamma = compute_gamma(x, gamma)
    K = np.zeros((x.shape[0], x.shape[0]))
    for i in range(len(K)):
        for j in range(len(K[0])):
            K[i,j] = np.exp(-gamma * np.linalg.norm(x[i]-x[j])**2)
    return K

def rbf_one(xi, xj, gamma):
    return np.exp(-gamma * np.linalg.norm(xi-xj)**2)

def linear(x):
    K = np.zeros((x.shape[0], x.shape[0]))
    for i in range(len(K)):
        for j in range(len(K[0])):
            K[i,j] = x[i].dot(x[j])
    return K

def poly(x, gamma='scale', deg=3, coef=0.0):
    if isinstance(gamma, str):
        gamma = compute_gamma(x, gamma)
    K = np.zeros((x.shape[0], x.shape[0]))
    for i in range(len(K)):
        for j in range(len(K[0])):
            K[i,j] = (gamma * x[i].dot(x[j]) + coef) ** deg
    return K

def sigmoid(x, gamma='scale', coef=0.0):
    if isinstance(gamma, str):
        gamma = compute_gamma(x, gamma)
    K = np.zeros((x.shape[0], x.shape[0]))
    for i in range(len(K)):
        for j in range(len(K[0])):
            K[i,j] = np.tanh(gamma*x[i].dot(x[j]) + coef)
    return K

if __name__ == "__main__":
    x = np.random.uniform(-1, 1, (10, 10))
    print("x: ", x)
    print("Linear: ", linear(x))
    print("RBF: ", rbf(x))
    print("Poly: ", poly(x))
    print("Sigmoid: ", sigmoid(x))