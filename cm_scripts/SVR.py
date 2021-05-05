import numpy as np
import kernel
from sklearn.preprocessing import StandardScaler
from deflected_subgradient import solveDeflected

class SVR:
    def __init__(self, kernel, kernel_args, box=1.0, eps=0.1):
        self.kernel = kernel
        self.box = box
        self.eps = eps

        self.gamma  = kernel_args['gamma'] if 'gamma' in kernel_args else 'scale'
        self.degree = kernel_args['degree'] if 'degree' in kernel_args else 1
        self.coef   = kernel_args['coef'] if 'coef' in kernel_args else 0

    def fit(self, x, y, optim_args, beta_init=None, verbose_optim=True):
        self.x = x
        self.y = y

        sc_X = StandardScaler()
        sc_Y = StandardScaler()
        self.xs = sc_X.fit_transform(self.x)
        self.ys = sc_Y.fit_transform(self.y)
        self.x_scaler = sc_X #TODO remove in case not nützlich
        self.y_scaler = sc_Y

        self.K = kernel.get_kernel(self)
        beta_init = np.zeros(self.x.shape) if beta_init is None else beta_init
        # var_esp for deflected should be initialized to self.esp value (?)
        self.beta, self.status = solveDeflected(beta_init, self.ys, self.K, self.box, optim_args=optim_args, verbose=verbose_optim)
        self.compute_sv()
        if self.kernel == "linear":
            self.W = np.dot(self.betasv.T, self.sv)
        # ------------------END------------------ #

    def compute_sv(self):
        mask = np.logical_or(self.beta > 1e-6, self.beta < -1e-6)
        support = np.vstack(np.vstack(np.arange(len(self.beta)))[mask])
        self.sv = np.vstack(self.xs[mask])
        y_sv = np.vstack(self.ys[mask])
        self.betasv = np.vstack(self.beta[mask])
        self.intercept = 0
        for i in range(self.betasv.size):
            self.intercept += y_sv[i]
            for j in range(self.beta.size):
                self.intercept -= self.beta[j] * self.K[j, support[i]]
        self.intercept /= self.betasv.size # average bias
        self.intercept -= self.eps # -eps -eps
    
    def predict(self, x):
        if x.size == 1:
            x = np.array([[x]])
        x = self.x_scaler.transform(x)
        
        if self.kernel == 'linear':
            return np.dot(self.W.T, x) + self.intercept

        gamma = 1/(self.sv.shape[1]*self.sv.var())
        if self.kernel == 'rbf':
            K = kernel.rbf(self.sv, x, gamma)
        elif self.kernel == 'poly':
            K = kernel.poly(self.sv, x, gamma, self.degree, self.coef)
        elif self.kernel == 'sigmoid':
            K = kernel.sigmoid(self.sv, x, gamma, self.coef)
        return self.y_scaler.inverse_transform(np.dot(self.beta.T, K) + self.intercept)

    def eps_ins_loss(self, y_pred):
        loss = 0
        for i in range(len(self.y)):
            loss += (abs(self.y[i]-y_pred[i]) - self.eps)**2 if abs(self.y[i]-y_pred[i]) > self.eps else 0
        return loss
