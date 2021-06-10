import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import time

from SVR import SVR
import kernel as k

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
            self.eps = param["eps"]
        if "optiargs" in param:
            self.opti_args = param["optiargs"]

    def run(self, train_x, train_output, test_x, test_output):
        # declare all SVR
        print("(GS - SVR) - Creating models")        
        models_conf = []
        kernel_conf = []
        precomp_kernels = {}
        for i, kernel in enumerate(self.kernel):
            for box in self.box:
                for eps in self.eps:
                    for _ in range(len(self.opti_args)):
                        models_conf.append(SVR(kernel, self.k_params[i], box, eps))
                        kernel_conf.append(i) # to get correct kernel afterwards

            # precompute kernels
            temp_model = SVR(kernel,self.k_params[i])
            temp_model.x, temp_model.xs = train_x, train_x
            precomp_kernel, precomp_gamma_value = k.get_kernel(temp_model)
            precomp_kernels[i] = (precomp_kernel, precomp_gamma_value)
        
        print(f"(GS - SVR) - Fitting {len(models_conf)} models")
        start_fit = time.time()
        for i, model in enumerate(models_conf):
            print(f"(GS - SVR) - model {i+1}/{len(models_conf)}", sep=" ")
            model.fit(train_x, train_output, self.opti_args[i%len(self.opti_args)], verbose_optim=False, precomp_kernel=precomp_kernels[kernel_conf[i]])
            print(f"\t(GS - SVR) - Time taken: {time.time() - start_fit} - Remaining: {(time.time() - start_fit) / (i+1) * (len(models_conf)-i-1)}")
        
        print("(GS - SVR) - Testing models")
        models_pred = []
        for i, model in enumerate(models_conf):
            tmp_pred = []
            for test in test_x:
                prediction = model.predict(test)
                tmp_pred.append(prediction)
            models_pred.append(tmp_pred)

        models_rmse = []
        for i, pred in enumerate(models_pred):
            error = 0
            for j, test_pred in enumerate(pred):
                print(test_output[j], test_pred)
                error += math.sqrt((test_output[j] - test_pred)**2)
            models_rmse.append(error/len(test_output))

        for i, error in enumerate(models_rmse):
            print(f"(GS - SVR) - SVR: {i} - RMSE {error} - PRED: {models_pred[i]} - MODEL: {models_conf[i]}")

        return models_conf[np.argmin(models_rmse)]

    def get_model_perturbations(self, model, n_perturbations, n_optimargs):
        kernel = []
        kparam = []
        optiargs = []

        # keep original model
        kernel.append(model.kernel)
        kparam.append({"gamma": model.gamma_value, "degree": model.degree, "coef": model.coef})
        optiargs.append(model.optim_args)

        # create perturbations of original
        gamma_perturbation = 0.2
        coef_perturbation = 1.0
        eps_perturbation = 10
        vareps_perturbation = 0.1
        for i in range(n_perturbations-1):
            if model.kernel == 'rbf':
                kernel.append('rbf')
                kparam.append({"gamma": np.random.uniform(model.gamma_value - gamma_perturbation, model.gamma_value + gamma_perturbation)})
            elif model.kernel == 'poly':
                kernel.append('poly')
                kparam.append({"gamma": np.random.uniform(model.gamma_value - gamma_perturbation, model.gamma_value + gamma_perturbation), "degree": model.degree, "coef": np.random.uniform(model.coef - coef_perturbation, model.coef + coef_perturbation)})
            elif model.kernel == 'sigmoid':
                kparam.append({"gamma": np.random.uniform(model.gamma_value - gamma_perturbation, model.gamma_value + gamma_perturbation), "coef": np.random.uniform(model.coef - coef_perturbation, model.coef + coef_perturbation)})
            else: # linear
                kernel.append('linear')
                kparam.append({})

        for i in range(n_optimargs-1):
            temp_optiargs = {}
            if 'eps' in model.optim_args:
                temp_optiargs['eps'] = np.random.uniform(model.optim_args['eps'] / eps_perturbation, model.optim_args['eps'] * eps_perturbation)
            if 'vareps' in model.optim_args:
                temp_optiargs['vareps'] = np.random.uniform(model.optim_args['vareps'] - vareps_perturbation, model.optim_args['vareps'] + vareps_perturbation)
            if 'maxiter' in model.optim_args:
                temp_optiargs['maxiter'] = model.optim_args['maxiter']
            optiargs.append(temp_optiargs)

        return kernel, kparam, optiargs