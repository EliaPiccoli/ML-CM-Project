import os
import numpy as np
import math
import time
import get_cup_dataset as dt
import pickle
import copy

from SVR import SVR

class Gridsearch():
    """
    Handler of the gridsearch setup and run. Constructed to behave as grid search on algorithmic specific parameters.
    """
    def __init__(self):
        """
        Initialize the model specific parameters containers:
        - kernel to utilize together with its parameters (gamma, degree, coef0)
        - box and epsilon tube 
        initialize the algorithm specific parameters containers:
        - vareps      : radius of epsilon-tube
        - maxiter     : maximum number of iterations
        - deltares    : reset value for delta
        - rho         : discount factor for delta
        - eps         : minimum relative value for the displacement of delta
        - alpha       : deflection coefficient                                [ alpha in (0,1)]
        - psi         : discount factor for the stepsize                      [ psi <= alpha]
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

    def run(self, inp, out, target_func_value=None, max_error_target_func_value=None, n_best=1, convergence_verbose=False):
        """
        Function to effectively run the GridSearch, returns top n performing models configuration.
        The performance is evaluated on reaching the lowest possible minimum (algorithmic aim).
        Args:
            inp (tensor): input data
            out (tensor): output data
            target_func_value (float, optional): necessary if 'accepted' convergence condition is wanted. Defaults to None.
            max_error_target_func_value (float, optional): range of error around target_func_value to define 'accepted' convergence condition. Defaults to None.
            n_best (int): number of best models configurations to return
            convergence_verbose (bool, optional): if set to True then at every model fitting end there will be plots on convergence rate and logarithmic residual error. Defaults to False.

        Returns:
            list(SVR): best performing models configurations
        """
        # check and set possible undeclared parameters about objective target
        if target_func_value is None:
            target_func_value = {'linear':-math.inf, 'rbf':-math.inf, 'poly':-math.inf}
        if max_error_target_func_value is None:
            max_error_target_func_value = 1e-3
        # declare all SVR
        print("(GS - SVR) - Creating models")        
        models_conf = []
        kernel_conf = []
        for i, kernel in enumerate(self.kernel):
            for box in self.box:
                for eps in self.eps:
                    for _ in range(len(self.opti_args)):
                        models_conf.append(SVR(kernel, self.k_params[i], box, eps))
                        kernel_conf.append(i) # to get correct kernel afterwards
        
        print(f"(GS - SVR) - Fitting {len(models_conf)} models")
        start_fit = time.time()
        f_bests = np.zeros(len(models_conf))
        # copy and delete models after fitting to avoid RAM overflow
        for i, model in enumerate(models_conf):
            print(f"(GS - SVR) - model {i+1}/{len(models_conf)}", sep=" ")
            copied_model = copy.deepcopy(model)
            copied_model.fit(inp, out, self.opti_args[i%len(self.opti_args)], target_func_value=target_func_value[model.kernel], max_error_target_func_value=max_error_target_func_value, optim_verbose=False, convergence_verbose=convergence_verbose)
            print("_"*100)
            print(f"\n\t(GS - SVR) - Time taken: {time.time() - start_fit} - Remaining: {(time.time() - start_fit) / (i+1) * (len(models_conf)-i-1)}")
            print(f"(GS - SVR) - SVR: {i} \nEXIT_STATUS: {copied_model.status} - F_BEST: {copied_model.history['fstar']} \nMODEL_OPTIM_ARGS: {copied_model.optim_args} \nMODEL_KERNEL(name/gamma/degree/coef0): {copied_model.kernel} {copied_model.gamma_value}/{copied_model.degree}/{copied_model.coef} \nMODEL_BOX: {copied_model.box}\n")
            f_bests[i] = copied_model.history['fstar']
            del copied_model
        
        # check if the number of requested models is valid
        n_best = n_best if n_best <= len(models_conf) else len(models_conf)

        # get the n_best models with the lowest f_best, print them, return their configuration
        best_indexes = np.argsort(f_bests)[:n_best]
        print("(GS - SVR) - Best configurations:", best_indexes, " with f_best ", np.sort(f_bests)[:n_best])
        return [models_conf[i] for i in best_indexes]
    
if __name__ == '__main__':
    start = time.time()
    # retrieve data to work with
    data, data_out = dt._get_cup('train')
    # get output_data as first output dimension
    data_out = data_out[:, 0]

    # setting the objective f_best, for each kernel fixed for a certain dimension (see line 126)
    target_func_value = {'linear':-math.inf, 'poly':-math.inf, 'rbf':-math.inf}

    gs = Gridsearch() 
    maxiter = 100000
    # get all the combinations using the requested algorithmic parameters
    optiargs = [{'alpha': 0.3, 'psi': 0.075, 'eps': 0.01, 'rho': 0.6, 'deltares': 0.001, 'maxiter': maxiter},
                {'alpha': 0.3, 'psi': 0.15, 'eps': 0.1, 'rho': 0.8, 'deltares': 0.0001, 'maxiter': maxiter},
                {'alpha': 0.3, 'psi': 0.15, 'eps': 0.01, 'rho': 0.9, 'deltares': 0.00001, 'maxiter': maxiter},
                {'alpha': 0.3, 'psi': 0.3, 'eps': 0.005, 'rho': 0.99, 'deltares': 0.000001, 'maxiter': maxiter},
                {'alpha': 0.5, 'psi': 0.075, 'eps': 0.1, 'rho': 0.6, 'deltares': 0.00001, 'maxiter': maxiter},

                {'alpha': 0.5, 'psi': 0.15, 'eps': 0.1, 'rho': 0.8, 'deltares': 0.00005, 'maxiter': maxiter},           # 6
                {'alpha': 0.5, 'psi': 0.3, 'eps': 0.1, 'rho': 0.9, 'deltares': 0.0001, 'maxiter': maxiter},             # 7
                
                {'alpha': 0.5, 'psi': 0.5, 'eps': 0.1, 'rho': 0.99, 'deltares': 0.001, 'maxiter': maxiter},
                {'alpha': 0.5, 'psi': 0.075, 'eps': 0.01, 'rho': 0.6, 'deltares': 0.00001, 'maxiter': maxiter},
                {'alpha': 0.5, 'psi': 0.15, 'eps': 0.01, 'rho': 0.8, 'deltares': 0.00005, 'maxiter': maxiter},
                {'alpha': 0.5, 'psi': 0.3, 'eps': 0.01, 'rho': 0.9, 'deltares': 0.0001, 'maxiter': maxiter},
                {'alpha': 0.5, 'psi': 0.5, 'eps': 0.01, 'rho': 0.99, 'deltares': 0.001, 'maxiter': maxiter},
                {'alpha': 0.5, 'psi': 0.15, 'eps': 0.005, 'rho': 0.6, 'deltares': 0.00001, 'maxiter': maxiter},
                {'alpha': 0.5, 'psi': 0.3, 'eps': 0.005, 'rho': 0.99, 'deltares': 0.001, 'maxiter': maxiter},

                {'alpha': 0.7, 'psi': 0.15, 'eps': 0.1, 'rho': 0.6, 'deltares': 0.00001, 'maxiter': maxiter},           # 15
                {'alpha': 0.7, 'psi': 0.3, 'eps': 0.1, 'rho': 0.8, 'deltares': 0.00005, 'maxiter': maxiter},            # 16
                {'alpha': 0.7, 'psi': 0.5, 'eps': 0.1, 'rho': 0.9, 'deltares': 0.0001, 'maxiter': maxiter},             # 17
                
                {'alpha': 0.7, 'psi': 0.7, 'eps': 0.1, 'rho': 0.99, 'deltares': 0.001, 'maxiter': maxiter},
                {'alpha': 0.7, 'psi': 0.15, 'eps': 0.01, 'rho': 0.6, 'deltares': 0.00001, 'maxiter': maxiter},
                {'alpha': 0.7, 'psi': 0.3, 'eps': 0.01, 'rho': 0.8, 'deltares': 0.00005, 'maxiter': maxiter},
                {'alpha': 0.7, 'psi': 0.5, 'eps': 0.01, 'rho': 0.9, 'deltares': 0.0001, 'maxiter': maxiter},
                {'alpha': 0.7, 'psi': 0.7, 'eps': 0.01, 'rho': 0.99, 'deltares': 0.001, 'maxiter': maxiter},
                {'alpha': 0.7, 'psi': 0.15, 'eps': 0.005, 'rho': 0.6, 'deltares': 0.00001, 'maxiter': maxiter},
                {'alpha': 0.7, 'psi': 0.7, 'eps': 0.005, 'rho': 0.99, 'deltares': 0.001, 'maxiter': maxiter}]
    gs.set_parameters(
        kernel=["linear"],
        kparam=[{}],
        box=[1], # taken from champion of previous analysis for linear kernel
        eps=[1], # taken from champion of previous analysis for linear kernel
        optiargs=optiargs
    )

    # run grid search, saving best configurations
    best_models_configurations = gs.run(
        data, data_out, target_func_value=target_func_value, n_best=5
    )

    # save best models configuration to output file
    save_path = os.path.dirname(__file__) + "/gs_models/gs_out"
    with open(save_path, "wb") as f:
        pickle.dump({"models": best_models_configurations}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"GridSearch output configuration succesfully saved to {save_path}")
