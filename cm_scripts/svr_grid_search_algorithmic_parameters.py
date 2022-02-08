import os
import numpy as np
import math
import time
import get_cup_dataset as dt
import pickle
import copy

from SVR import SVR
import kernel as k

class Gridsearch():
    """
    handler of the gridsearch setup and run
    """
    def __init__(self):
        """
        initialize the model specific parameters containers:
        - kernel to utilize together with its parameters (gamma, degree, coef0)
        - box and epsilon tube 
        initialize the algorithm specific parameters containers:
        -  vareps      : radius of epsilon-tube
        - maxiter     : maximum number of iterations
        - deltares    : reset value for delta                                 [ used if we find a better point than the current estimate ]
        - rho         : discount factor for delta                             [ used if we don't find a better point ]
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
        setup the grid search parameters. kernel and kparam have to be the same size.
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
        function to effectively run the GridSearch, returns top n performing models. the performance is evaluated on reaching the lowest possible minimum.
        """
        # check and set possible undeclared parameters about objective target
        if target_func_value is None:
            target_func_value = {'linear':math.inf, 'rbf':math.inf, 'poly':math.inf}
        if max_error_target_func_value is None:
            max_error_target_func_value = 1e-3

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
        
        print(f"(GS - SVR) - Fitting {len(models_conf)} models")
        start_fit = time.time()
        f_bests = np.zeros(len(models_conf))
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

        # get the n_best models with the lowest f_best, print them, return them
        best_indexes = np.argsort(f_bests)[:n_best]
        print("(GS - SVR) - Best configurations:", best_indexes, " with f_best ", np.sort(f_bests)[:n_best])
        return [models_conf[i] for i in best_indexes]
    
def create_algorithmic_configurations(alphas=['dc'],  epses=['dc'], rhos=['dc'], deltareses=['dc']): 
    """
    function to create the algorithmic parametered configurations, implementing a cartesian between the input features
    """
    configs = []
    for alpha in alphas:
        psis = [alpha/4, alpha/2, alpha] if not isinstance(alpha, str) else ['dc']
        for psi in psis:
            for eps in epses:
                for rho in rhos:
                    for deltares in deltareses:
                        configs.append({})
                        if not isinstance(alpha, str):
                            configs[-1]['alpha'] = alpha
                        if not isinstance(psi, str):
                            configs[-1]['psi'] = psi
                        if not isinstance(eps, str):
                            configs[-1]['eps'] = eps
                        if not isinstance(rho, str):
                            configs[-1]['rho'] = rho
                        if not isinstance(deltares, str):
                            configs[-1]['deltares'] = deltares
    # TODO comment these two lines to set the maxiter to the very high default value
    for i in range(len(configs)):
        configs[i]['maxiter'] = 3

    return configs

if __name__ == '__main__':
    start = time.time()
    first_dim = True
    data, data_out = dt._get_cup('train')
    data_out = data_out[:, 0]

    # setting the objective targets, divided by kernel, gotten with sklearn
    target_func_value = {'linear':-3138.9592, 'poly':-22463.8118, 'rbf':-18917.8941}
    """
    # the old way
    gs = Gridsearch()
    # create all the combinations using the requested algorithmic parameters
    optiargs = create_algorithmic_configurations(alphas=[0.3, 0.5, 0.7], epses=[0.1, 0.01, 0.005], rhos=[0.6, 0.8, 0.9], deltareses=[1e-3, 1e-4, 1e-5])
    # optiargs = create_algorithmic_configurations(alphas=[0.3], epses=[0.1], rhos=[0.6], deltareses=[1e-3])  # for testing everything works
    print(optiargs)
    gs.set_parameters(
        kernel=["linear"],
        kparam=[{}],
        box=[1], # taken from champion of previous analysis
        eps=[1], # taken from champion of previous analysis
        optiargs=optiargs
    )
    best_models = gs.run(
        data, data_out, target_func_value=target_func_value, n_best=5
    )

    # save best models to output file
    save_path = os.path.dirname(__file__) + "/gs_models/gs_out"
    with open(save_path, "wb") as f:
        pickle.dump({"models": best_models}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"GridSearch output succesfully saved to {save_path}")
    """

    # the new, beautiful, fast, ep no rabia way
    gs = Gridsearch() 
    maxiter = 3 # 100000
    # get all the combinations using the requested algorithmic parameters
    optiargs = [{'alpha': 0.3, 'psi': 0.075, 'eps': 0.01, 'rho': 0.6, 'deltares': 0.001, 'maxiter': maxiter},
                {'alpha': 0.3, 'psi': 0.15, 'eps': 0.1, 'rho': 0.8, 'deltares': 0.0001, 'maxiter': maxiter},
                {'alpha': 0.3, 'psi': 0.15, 'eps': 0.01, 'rho': 0.9, 'deltares': 0.00001, 'maxiter': maxiter},
                {'alpha': 0.3, 'psi': 0.3, 'eps': 0.005, 'rho': 0.99, 'deltares': 0.000001, 'maxiter': maxiter},

                {'alpha': 0.5, 'psi': 0.075, 'eps': 0.1, 'rho': 0.6, 'deltares': 0.00001, 'maxiter': maxiter},
                # top 
                {'alpha': 0.5, 'psi': 0.15, 'eps': 0.1, 'rho': 0.8, 'deltares': 0.00005, 'maxiter': maxiter},
                {'alpha': 0.5, 'psi': 0.3, 'eps': 0.1, 'rho': 0.9, 'deltares': 0.0001, 'maxiter': maxiter},
                
                {'alpha': 0.5, 'psi': 0.5, 'eps': 0.1, 'rho': 0.99, 'deltares': 0.001, 'maxiter': maxiter},
                {'alpha': 0.5, 'psi': 0.075, 'eps': 0.01, 'rho': 0.6, 'deltares': 0.00001, 'maxiter': maxiter},
                {'alpha': 0.5, 'psi': 0.15, 'eps': 0.01, 'rho': 0.8, 'deltares': 0.00005, 'maxiter': maxiter},
                {'alpha': 0.5, 'psi': 0.3, 'eps': 0.01, 'rho': 0.9, 'deltares': 0.0001, 'maxiter': maxiter},
                {'alpha': 0.5, 'psi': 0.5, 'eps': 0.01, 'rho': 0.99, 'deltares': 0.001, 'maxiter': maxiter},
                {'alpha': 0.5, 'psi': 0.15, 'eps': 0.005, 'rho': 0.6, 'deltares': 0.00001, 'maxiter': maxiter},
                {'alpha': 0.5, 'psi': 0.3, 'eps': 0.005, 'rho': 0.99, 'deltares': 0.001, 'maxiter': maxiter},

                # top
                {'alpha': 0.7, 'psi': 0.15, 'eps': 0.1, 'rho': 0.6, 'deltares': 0.00001, 'maxiter': maxiter},
                {'alpha': 0.7, 'psi': 0.3, 'eps': 0.1, 'rho': 0.8, 'deltares': 0.00005, 'maxiter': maxiter},
                {'alpha': 0.7, 'psi': 0.5, 'eps': 0.1, 'rho': 0.9, 'deltares': 0.0001, 'maxiter': maxiter},
                
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
        box=[1], # taken from champion of previous analysis
        eps=[1], # taken from champion of previous analysis
        optiargs=optiargs
    )
    best_models = gs.run(
        data, data_out, target_func_value=target_func_value, n_best=5
    )

    # save best models to output file
    save_path = os.path.dirname(__file__) + "/gs_models/gs_out"
    with open(save_path, "wb") as f:
        pickle.dump({"models": best_models}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"GridSearch output succesfully saved to {save_path}")
    