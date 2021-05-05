import numpy as np
import copy
import math

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
        print("Fitting models")
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

        models_mee = []
        for i, pred in enumerate(models_pred):
            error = 0
            for j, test_pred in enumerate(pred):
                error += math.sqrt((test_output[j] - test_pred)**2)
            models_mee.append(error/len(test_output))

        for i, error in enumerate(models_mee):
            print(f"SVR: {i} - RMSE {error} - PRED: {models_pred[i]}")

        # TODO: ml cup has 2 ouputs -> 2 SVR, avg the error over the single episodes or the total (?)

if __name__ == "__main__":
    x = np.vstack(np.arange(-10,11,1))
    degree = 2
    noising_factor = 0.1
    y = [xi**degree for xi in x]
    y = [ yi + noising_factor * (np.random.rand()*yi) for yi in y]
    y = np.array(y)

    test_x = [12, 15, 18]
    test_y = [144, 225, 324]

    gs = Gridsearch()
    gs.set_parameters(
        kernel=["linear", "rbf", "poly"],
        kparam=[{}, {"gamma":"scale"}, {"degree":2, "gamma":"auto"}],
        optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':1e-4, 'maxiter':1e3}]
    )
    gs.run(
        x, y, test_x, test_y
    )