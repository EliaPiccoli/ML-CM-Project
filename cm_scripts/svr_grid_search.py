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
# - delta_res (?)
# - rho (?)
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

    def run(self, train_x, train_output, test_X, test_output):
        # declare all SVR        
        models_conf = []
        for i, kernel in enumerate(self.kernel):
            for box in self.box:
                for eps in self.eps:
                    models_conf.append(SVR(kernel, self.k_params[i], box, eps))
        
        # fit
        for i, model in enumerate(models_conf): # parallelizable
            model.fit(train_x, train_output, self.opti_args[i], verbose_optim=False)
        
        # test
        models_pred = []
        for i, model in models_conf: # parallelizable
            tmp_pred = []
            for test in test_X:
                tmp_pred.append(model.predict(test))
            models_pred.append(tmp_pred)
        
        # compare
        models_mee = []
        for i, pred in models_pred:
            error = 0
            for j, test_pred in enumerate(pred):
                error += math.sqrt((test_output[j] - test_pred)**2)
            models_mee.append(error/len(test_output))

        # TODO: ml cup has 2 ouputs -> 2 SVR, avg the error over the single episodes or the total (?)

if __name__ == "__main__":
    # read dataset
    # ...

    gs = Gridsearch()
    gs.set_parameters(
        # ...
    )
    gs.run(
        # ...
    )