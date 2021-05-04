import numpy as np
import kernel
from sklearn.preprocessing import StandardScaler

class SVR:
    def __init__(self, kernel, box=1.0, eps=0.1):
        self.kernel = kernel
        self.box = box
        self.eps = eps

    def fit(self, x, y, kernel_args, optim_args, beta_init=None, verbose_optim=True):
        self.x = x
        self.y = y

        self.gamma  = kernel_args['gamma'] if 'gamma' in kernel_args else 'scale'
        self.degree = kernel_args['degree'] if 'degree' in kernel_args else 1
        self.coef   = kernel_args['coef'] if 'coef' in kernel_args else 0

        sc_X = StandardScaler()
        sc_y = StandardScaler()
        self.x_scaled = sc_X.fit_transform(self.x)
        self.y_scaled = sc_Y.fit_transform(self.y)
        self.x_scaler = sc_X
        self.y_scaler = sc_Y

        self.K = kernel.get_kernel(self.x_scaled, self.kernel, self.x, self.gamma, self.degree, self.coef)

        beta_init = np.zeros(self.x.shape) if beta_init is None else beta_init
        self.beta = solveDeflected(beta_init, self.y_scaled, self.K, self.box, optim_args=optim_args, verbose=verbose_optim)

def eps_ins_loss(y_true, y_pred,eps=0.1):
    loss = 0
    for i in range(len(y_true)):
        temp_loss = abs(y_true[i]-y_pred[i])
        if temp_loss > eps:
            loss += (temp_loss - eps)**2
    return loss

def predict_linear(W, b, x):
    return np.dot(np.transpose(W), x) + b

def predict_rbf(b, beta, x, sv, gamma=None):
    if gamma is None:
        gamma = 1/(sv.shape[1]*sv.var())
    K = np.zeros((sv.shape[0], x.shape[0]))
    for i in range(len(K)):
        for j in range(len(K[0])):
            K[i,j] = np.exp(-gamma * np.linalg.norm(sv[i]-x[j])**2)
    return np.dot(np.transpose(beta), K) + b

def predict_poly(b, beta, x, sv, deg, gamma=None):
    if gamma is None:
        gamma = 1/(sv.shape[1]*sv.var())
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