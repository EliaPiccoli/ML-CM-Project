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

    def run(self, train_x, train_output, val_x, val_output, convergence_verbose=False):
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
            model.fit(train_x, train_output, self.opti_args[i%len(self.opti_args)], optim_verbose=False, precomp_kernel=precomp_kernels[kernel_conf[i]], convergence_verbose=convergence_verbose)
            print(f"\t(GS - SVR) - Time taken: {time.time() - start_fit} - Remaining: {(time.time() - start_fit) / (i+1) * (len(models_conf)-i-1)}")
        
        print("(GS - SVR) - Evaluating models")
        models_predt = []
        for i, model in enumerate(models_conf):
            tmp_pred = []
            for inp in train_x:
                prediction = model.predict(inp)
                tmp_pred.append(prediction)
            models_predt.append(tmp_pred)

        models_meet = []
        for i, pred in enumerate(models_predt):
            error = 0
            for j, train_pred in enumerate(pred):
                # print(train_output[j], train_pred)
                error += math.sqrt((train_output[j] - train_pred)**2)
            models_meet.append(error/len(train_output))

        models_pred = []
        for i, model in enumerate(models_conf):
            tmp_pred = []
            for inp in val_x:
                prediction = model.predict(inp)
                tmp_pred.append(prediction)
            models_pred.append(tmp_pred)

        models_mee = []
        for i, pred in enumerate(models_pred):
            error = 0
            for j, val_pred in enumerate(pred):
                error += math.sqrt((val_output[j] - val_pred)**2)
            models_mee.append(error/len(val_output))

        for i in range(len(models_mee)):
            print(f"(GS - SVR) - SVR: {i} - TR MEE {models_meet[i]} - VL MEE {models_mee[i]} - MODEL: {models_conf[i]}\n")

        index = np.argmin(models_mee)
        print("(GS - SVR) - Best configuration:", index)
        return models_conf[index]

    def get_model_perturbations(self, model, n_perturbations, n_optimargs, n_box_perturb=1):
        kernel = []
        kparam = []
        optiargs = []
        eps=[]
        box=[]

        # keep original model
        kernel.append(model.kernel)
        kparam.append({"gamma": model.gamma_value, "degree": model.degree, "coef": model.coef})
        optiargs.append(model.optim_args)
        eps.append(model.eps)
        box.append(model.box)

        # create perturbations of original
        gamma_perturbation = 0.2
        coef_perturbation = 1.0
        eps_perturbation = 10
        box_perturbation = 10
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
            if 'maxiter' in model.optim_args:
                temp_optiargs['maxiter'] = model.optim_args['maxiter']
            temp_optiargs['vareps'] = model.eps
            optiargs.append(temp_optiargs)

        for i in range(n_box_perturb-1):
            box.append(model.box + np.random.uniform(-model.box/box_perturbation, model.box/box_perturbation))

        return kernel, kparam, optiargs, eps, box