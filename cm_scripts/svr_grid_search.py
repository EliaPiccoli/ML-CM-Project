import numpy as np
import math
import time
import kernel as k
from SVR import SVR

class Gridsearch():
    """class constructed to behave as grid search on model parameters.
    """    
    def __init__(self):
        """initialize grid search to a minimal search space of one case
        """        
        self.kernel = ['rbf']
        self.k_params = [{'gamma':'scale'}]
        self.box = [1]
        self.eps = [0.1]
        self.opti_args = [{}]

    def set_parameters(self, **param):
        """
        set grid search parameters given a dictionary of parameters param. Setup the grid search parameters. kernel and kparam have to be the same size.
        """        
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
        """run grid search, returning best performing model based on MEE

        Args:
            train_x (tensor): input training data
            train_output (tensor): output training data
            val_x (tensor): input validation data (model selection)
            val_output (tensor): output validation data (model selection)
            convergence_verbose (bool, optional): if set to True then at every model fitting end there will be plots on convergence rate and logarithmic residual error. Defaults to False.

        Returns:
            SVR: best performing model
        """        
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
                        kernel_conf.append(i) # keep model index in order to get correct kernel afterwards

            # precompute kernels (many configurations may share the same kernel)
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

        # get models predictions on training data
        models_predt = []
        for i, model in enumerate(models_conf):
            tmp_pred = []
            for inp in train_x:
                prediction = model.predict(inp)
                tmp_pred.append(prediction)
            models_predt.append(tmp_pred)

        # compute training MEE for all models
        models_meet = []
        for i, pred in enumerate(models_predt):
            error = 0
            for j, train_pred in enumerate(pred):
                error += math.sqrt((train_output[j] - train_pred)**2)
            models_meet.append(error/len(train_output))

        # get models predictions on validation data
        models_pred = []
        for i, model in enumerate(models_conf):
            tmp_pred = []
            for inp in val_x:
                prediction = model.predict(inp)
                tmp_pred.append(prediction)
            models_pred.append(tmp_pred)

        # compute validation MEE for all models
        models_mee = []
        for i, pred in enumerate(models_pred):
            error = 0
            for j, val_pred in enumerate(pred):
                error += math.sqrt((val_output[j] - val_pred)**2)
            models_mee.append(error/len(val_output))

        # print out results
        for i in range(len(models_mee)):
            print(f"(GS - SVR) - SVR: {i} - TR MEE {models_meet[i]} - VL MEE {models_mee[i]} - MODEL: {models_conf[i]}\n")

        # get best performing model on validation set, return it
        index = np.argmin(models_mee)
        print("(GS - SVR) - Best configuration:", index)
        return models_conf[index]

    def get_model_perturbations(self, model, n_perturbations, n_optimargs, n_box_perturb=1):
        """function to create perturbated configurations. Useful for 'fine grid search'

        Args:
            model (SVR): original model, for original configurations
            n_perturbations (int): define number of perturbations to create kernel-wise
            n_optimargs (int): define number of perturbations to create algorithm-wise
            n_box_perturb (int, optional): define number of perturbations to create box-wise. Defaults to 1.

        Returns:
            list: return all perturbations in a list ready for calling the 'run' method
        """        
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

        # set (empirically) ranges of perturbation for the parameters
        gamma_perturbation = 0.2
        coef_perturbation = 1.0
        eps_perturbation = 10
        box_perturbation = 10
        # create perturbations of original (kernel)
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
        # create perturbations of original (algorithm)
        for i in range(n_optimargs-1):
            temp_optiargs = {}
            if 'eps' in model.optim_args:
                temp_optiargs['eps'] = np.random.uniform(model.optim_args['eps'] / eps_perturbation, model.optim_args['eps'] * eps_perturbation)
            if 'maxiter' in model.optim_args:
                temp_optiargs['maxiter'] = model.optim_args['maxiter']
            temp_optiargs['vareps'] = model.eps
            optiargs.append(temp_optiargs)
        # create perturbations of original (box)
        for i in range(n_box_perturb-1):
            box.append(model.box + np.random.uniform(-model.box/box_perturbation, model.box/box_perturbation))

        return kernel, kparam, optiargs, eps, box