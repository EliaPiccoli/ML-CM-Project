import numpy as np

def predict_linear(W, b, x):
    return np.dot(np.transpose(W), x) + b

def predict_rbf(W, b, beta, x, sv):
    gamma = 1/(sv.shape[0]*sv.var())
    K = np.zeros((sv.shape[0], x.shape[0]))
    for i in range(len(K)):
        for j in range(len(K[0])):
            K[i,j] = np.exp(-gamma * np.linalg.norm(sv[i]-x[j])**2)
    # print(K)
    return np.dot(np.transpose(beta), K) + b

def predict_poly(W, b, beta, x, sv, deg):
    gamma = 1/(sv.shape[0]*sv.var())
    K = np.zeros((sv.shape[0], x.shape[0]))
    for i in range(len(K)):
        for j in range(len(K[0])):
            K[i,j] = (gamma * sv[i].dot(x[j]) + 0.0) ** deg
    return np.dot(np.transpose(beta), K) + b






# mask = np.logical_or(beta > 1e-6, beta < -1e-6)
# support = np.vstack(np.vstack(np.arange(len(beta)))[mask])
# suppvect = np.vstack(x[mask])
# y_sv = np.vstack(y[mask])
# beta = np.vstack(beta[mask])

# # only for linear kernel
# W = np.dot(np.transpose(beta), suppvect)

# b = 0
# for i in range(beta.size):
#     b += y_sv[i]
#     b -= np.sum(beta * K[support[i], np.hstack(mask)])
# b -= vareps
# b /= len(beta)
# print(f"W : {W} - b: {b}")