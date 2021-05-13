import numpy as np
import kernel
from sklearn.preprocessing import StandardScaler
from deflected_subgradient import solveDeflected
import matplotlib.pyplot as plt

class SVR:
    def __init__(self, kernel, kernel_args={}, box=1.0, eps=0.1):
        self.kernel = kernel
        self.box = box
        self.eps = eps

        self.gamma  = kernel_args['gamma'] if 'gamma' in kernel_args else 'scale'
        self.degree = kernel_args['degree'] if 'degree' in kernel_args else 1
        self.coef   = kernel_args['coef'] if 'coef' in kernel_args else 0
        self.optim_args = None
    
    def __str__(self):
        model_as_string = '\n'
        model_as_string += "Kernel: "+self.kernel
        if self.optim_args is not None: # model fit has happened
            if self.kernel == 'linear':
                model_as_string += "\nW: "+str(self.W)
            elif self.kernel == 'rbf':
                model_as_string += "\nGamma: "+str(self.gamma_value)
            elif self.kernel == 'poly':
                model_as_string += "\nGamma: "+str(self.gamma_value)
                model_as_string += "\tDegree: "+str(self.degree)
                model_as_string += "\tCoef: "+str(self.coef)
            elif self.kernel == 'sigmoid':
                model_as_string += "\nGamma: "+str(self.gamma_value)
                model_as_string += "\tCoef: "+str(self.coef)
            model_as_string += "\nIntercept: "+str(self.intercept)
            model_as_string += "\Optim_args: "+str(self.optim_args)
        model_as_string += "\nBox: "+str(self.box)
        return model_as_string
        

            

    def fit(self, x, y, optim_args, scaled=False, beta_init=None, verbose_optim=True):
        self.x = x
        self.y = y
        self.optim_args = optim_args
        
        self.scaled = scaled
        if scaled:
            sc_X = StandardScaler()
            sc_Y = StandardScaler()
            self.xs = sc_X.fit_transform(self.x)
            self.ys = sc_Y.fit_transform(self.y)
            self.x_scaler = sc_X
            self.y_scaler = sc_Y
        else:
            self.xs = x
            self.ys = y

        self.K, self.gamma_value = kernel.get_kernel(self)
        print(self.gamma_value)
        beta_init = np.zeros(self.x.shape) if beta_init is None else beta_init
        optim_args['vareps'] = self.eps
        self.beta, self.status, self.betas_history = solveDeflected(beta_init, self.ys, self.K, self.box, optim_args=optim_args, verbose=verbose_optim)
        self.compute_sv()
        if self.kernel == "linear":
            self.W = np.dot(self.betasv.T, self.sv)
        # ------------------END------------------ #

    def plot_loss(self):
        temp_beta, temp_sv, temp_betasv, temp_intercept = self.beta, self.sv, self.betasv, self.intercept
        loss_vect = []
        for betas in self.betas_history:
            self.beta = betas
            flag = self.compute_sv(plotting=True)
            if not flag:
                continue
            y_pred = [float(self.predict(self.x[i].reshape(1,-1))) for i in range(self.x.size)]
            loss_vect.append(self.eps_ins_loss(y_pred))
        plt.plot(range(len(loss_vect)),loss_vect)
        plt.show()
        self.beta, self.sv, self.betasv, self.intercept = temp_beta, temp_sv, temp_betasv, temp_intercept

    def compute_sv(self, plotting=False):
        mask = np.logical_or(self.beta > 1e-8, self.beta < -1e-8)
        if True not in mask and plotting:
            return False
        else: # to avoid crashing the application
            mask = np.logical_or(self.beta == np.max(self.beta), self.beta == np.min(self.beta))
        support = np.vstack(np.vstack(np.arange(len(self.beta)))[mask])
        x_mask = np.repeat(mask, self.xs.shape[1], axis=1)
        self.sv = np.vstack(self.xs[x_mask])
        y_sv = np.vstack(self.ys.reshape(-1,1)[mask])
        self.betasv = np.vstack(self.beta[mask])
        self.intercept = 0
        for i in range(self.betasv.size):
            self.intercept += y_sv[i]
            for j in range(self.beta.size):
                self.intercept -= self.beta[j] * self.K[j, support[i]]
        self.intercept /= self.betasv.size # average bias
        self.intercept -= self.eps # -eps -eps
        return True
    
    def predict(self, x):
        if isinstance(x, int):
            x = np.array([[x]])
        if self.scaled:
            x = self.x_scaler.transform(x)
        
        if self.kernel == 'linear':
            prediction = np.dot(self.W.T, x) + self.intercept
            return prediction if not self.scaled else self.y_scaler.inverse_transform(prediction)

        if isinstance(self.gamma, str):
            gamma = 1/(self.sv.shape[1]*self.sv.var()) if self.gamma == "scale" else 1/(self.sv.shape[1])
        else:
            gamma = self.gamma
        if self.kernel == 'rbf':
            K, _ = kernel.rbf(self.sv, x, gamma)
        elif self.kernel == 'poly':
            K, _ = kernel.poly(self.sv, x, gamma, self.degree, self.coef)
        elif self.kernel == 'sigmoid':
            K, _ = kernel.sigmoid(self.sv, x, gamma, self.coef)
        prediction = np.dot(self.betasv.T, K) + self.intercept
        return prediction if not self.scaled else self.y_scaler.inverse_transform(prediction)

    def eps_ins_loss(self, y_pred):
        loss = 0
        for i in range(len(self.y)):
            loss += (abs(self.y[i]-y_pred[i]) - self.eps)**2 if abs(self.y[i]-y_pred[i]) > self.eps else 0
        return loss


if __name__ == "__main__":
    prova = 2
    if prova == 0:
        x = np.vstack(np.arange(-10,11,1))
        degree = 2
        noising_factor = 0.1
        y = [xi**degree for xi in x]
        y = [ yi + noising_factor * (np.random.rand()*yi) for yi in y]
        y = np.array(y)
        svr = SVR('poly', {'degree':2})
    elif prova == 1:
        x = np.vstack(np.arange(1,11,1))
        y = np.array([[45000],[50000],[60000],[80000],[110000],[150000],[200000],[300000],[500000],[1000000]], dtype=np.float64)
        svr = SVR("rbf", {"gamma": 10})
    elif prova == 2:
        x = np.vstack(np.arange(-50,51,1))
        degree = 2
        noising_factor = 0.1
        y = [xi**degree for xi in x]
        y = [ yi + noising_factor * (np.random.rand()*yi) for yi in y]
        y = np.array(y)
        svr = SVR('poly', {'degree':2, 'gamma':'auto'})

    svr.fit(x,y, {'eps':1e-2, 'maxiter':3e3}, scaled=False)
    svr.plot_loss()
    all_to_predict = [12,15,18]
    for to_predict in all_to_predict:
        pred = svr.predict(to_predict)
        print(f'PREDICTION (INPUT = {to_predict})', pred)
    print(f"b: {svr.intercept}")
    print(f"Gamma: {svr.gamma_value} - Box: {svr.box}")
    pred = [float(svr.predict(np.array([x[i]]))) for i in range(x.size)]
    print("LOSS:", svr.eps_ins_loss(pred))
    plt.scatter(x, y , color="red")
    plt.plot(x, pred, color="blue")
    plt.title('SVR')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.show()
    print(svr)