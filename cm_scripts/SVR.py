import numpy as np
import kernel

def eps_insensitive_quad_loss_funct(x,y,eps,b,kernel='rbf', W=None, beta=None, sv=None, gamma=None):
    loss = 0
    fx = np.array([])
    if kernel == 'linear' and W is not None:
        fx = W * x + b
    elif kernel == "rbf" and beta is not None and sv is not None:
        if gamma is None:
            gamma = 'scale'
        for xi in x:
            fx.append([predict_rbf(b, beta, xi, sv, gamma)])
    for i in range(len(y)):
        temp_loss = abs(y[i]-fx[i])
        if temp_loss > eps:
            loss += (temp_loss - eps)**2

def predict_linear(W, b, x):
    return np.dot(np.transpose(W), x) + b

def predict_rbf(b, beta, x, sv, xs=None, orig_beta=None, gamma=None):
    if gamma is None:
        gamma = 1/(sv.shape[0]*sv.var())
    K = np.zeros((sv.shape[0], x.shape[0]))
    for i in range(len(K)):
        for j in range(len(K[0])):
            K[i,j] = np.exp(-gamma * np.linalg.norm(sv[i]-x[j])**2)
    # print(K)
    fx = np.dot(np.transpose(beta), K) + b
    if xs is not None:
        fx = 0
        for betai,xi in zip(orig_beta, xs):
            fx += betai*kernel.rbf_one(xi[0],x[0],gamma)
        fx += b
        print("Difference between methods: ",np.dot(np.transpose(beta), K) + b - fx)
    return fx





def predict_poly(b, beta, x, sv, deg):
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