import numpy as np
import copy
import math
import matplotlib.pyplot as plt

from SVR import SVR

# Parametri SVR:
# - kernel
#   - gamma
#   - degree
#   - coef
# - box
# - eps
# Parametri Deflected:
# - iteration (?)
# - eps
# - alpha
# - psi

class Gridsearch():
    def __init__(self):
        self.kernel = ['rbf']
        self.k_params = [{'gamma':'scale'}]
        self.box = [1]
        self.eps = [0.1]
        self.opti_args = [{}]

    def set_parameters(self, **param):
        if "kernel" in param:
            self.kernel = param["kernel"]
        if "kparam" in param:
            self.k_params = param["kparam"]
        if "box" in param:
            self.box = param["box"]
        if "eps" in param:
            self.kernel = param["eps"]
        if "optiargs" in param:
            self.opti_args = param["optiargs"]

    def run(self, train_x, train_output, test_x, test_output):
        # declare all SVR
        print("Creating models")        
        models_conf = []
        for i, kernel in enumerate(self.kernel):
            for box in self.box:
                for eps in self.eps:
                    for _ in range(len(self.opti_args)):
                        models_conf.append(SVR(kernel, self.k_params[i], box, eps))
        
        # fit
        print(f"Fitting {len(models_conf)} models")
        for i, model in enumerate(models_conf): # parallelizable
            model.fit(train_x, train_output, self.opti_args[i%len(self.opti_args)], verbose_optim=False)
        
        # test
        print("Testing models")
        models_pred = []
        for i, model in enumerate(models_conf): # parallelizable
            tmp_pred = []
            for test in test_x:
                tmp_pred.append(model.predict(test))
            models_pred.append(tmp_pred)
        
        # compare: HOW (?)

        models_rmse = []
        for i, pred in enumerate(models_pred):
            error = 0
            for j, test_pred in enumerate(pred):
                error += math.sqrt((test_output[j] - test_pred)**2)
            models_rmse.append(error/len(test_output))

        for i, error in enumerate(models_rmse):
            print(f"SVR: {i} - RMSE {error} - PRED: {models_pred[i]} - MODEL: {models_conf[i]}")

        # TODO: ml cup has 2 ouputs -> 2 SVR, avg the error over the single episodes or the total (?)
        return models_conf[np.argmin(models_rmse)]

    def get_model_perturbations(self, model, n_perturbations, n_optimargs):
        kernel = []
        kparam = []
        optiargs = []
        gamma_perturbation = 0.2
        eps_perturbation = 10
        vareps_perturbation = 0.1
        for i in range(n_perturbations):
            if model.kernel == 'rbf':
                kernel.append('rbf')
                kparam.append({"gamma": np.random.uniform(model.gamma_value - gamma_perturbation, model.gamma_value + gamma_perturbation)})
            elif model.kernel == 'poly':
                kernel.append('poly')
                kparam.append({"gamma": np.random.uniform(model.gamma_value - gamma_perturbation, model.gamma_value + gamma_perturbation), "degree": model.degree, "coef": model.coef})
        for i in range(n_optimargs):
            temp_optiargs = {}
            if 'eps' in model.optim_args:
                temp_optiargs['eps'] = np.random.uniform(model.optim_args['eps'] / eps_perturbation, model.optim_args['eps'] * eps_perturbation)
            if 'vareps' in model.optim_args:
                temp_optiargs['vareps'] = np.random.uniform(model.optim_args['vareps'] - vareps_perturbation, model.optim_args['vareps'] + vareps_perturbation)
            if 'maxiter' in model.optim_args:
                temp_optiargs['maxiter'] = model.optim_args['maxiter']
            optiargs.append(temp_optiargs)
        return kernel, kparam, optiargs

if __name__ == "__main__":
    x = np.vstack(np.arange(-50,51,1))
    degree = 2
    noising_factor = 0.1
    y = [xi**degree for xi in x]
    # y = np.sin(x)
    # y = [ yi + noising_factor * (np.random.rand()*yi) for yi in y]
    y = np.array(y, dtype=np.float64)

    test_x = [12, 15, 18]
    test_y = [144, 225, 324]
    # test_y = [0.2079, 0.2588, 0.3090]

    gs = Gridsearch()
    gs.set_parameters(
        kernel=["linear", "rbf", "poly", "poly", "poly"],
        kparam=[{}, {"gamma":"scale"}, {"degree":2, "gamma":0.6}, {"degree":2, "gamma":"scale"}, {"degree":2, "gamma":"auto"}],
        optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':1e-3, 'maxiter':3e3}]
    )
    best_coarse_model = gs.run(
        x, y, test_x, test_y
    )
    print("BEST COARSE GRID SEARCH MODEL:",best_coarse_model)

    kernel, kparam, optiargs = gs.get_model_perturbations(best_coarse_model, 4, 4)
    print(kernel, kparam, optiargs)
    gs.set_parameters(
        kernel=kernel,
        kparam=kparam,
        optiargs=optiargs
    )
    best_fine_model = gs.run(
        x, y, test_x, test_y
    )
    print("BEST FINE GRID SEARCH MODEL:",best_fine_model)

    svr = best_fine_model
    to_predict = 12
    pred = svr.predict(to_predict)
    print(f"b: {svr.intercept}")
    print(f"Gamma: {svr.gamma_value} - Box: {svr.box}")
    print(f'PREDICTION (INPUT = {to_predict})', pred)
    pred = [float(svr.predict(np.array([x[i]]))) for i in range(x.size)]
    print("LOSS:", svr.eps_ins_loss(pred))
    plt.scatter(x, y , color="red")
    plt.plot(x, pred, color="blue")
    plt.title('SVR')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.show()