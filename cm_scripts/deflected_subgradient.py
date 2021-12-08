from kp import solveKP
import numpy as np
import copy
import math

def unrollArgs(optim_args):
    """
        vareps      : radius of epsilon-tube
        maxiter     : maximum number of iterations
        deltares    : reset value for delta                                 [ used if we find a better point than the current estimate ]
        rho         : discount factor for delta                             [ used if we don't find a better point ]
        eps         : minimum relative value for the displacement of delta
        alpha       : deflection coefficient                                [ alpha in (0,1)]
        psi         : discount factor for the stepsize                      [ psi <= alpha]
    """
    vareps = optim_args['vareps'] if 'vareps' in optim_args else 0.1
    maxiter = optim_args['maxiter'] if 'maxiter' in optim_args else 1e6  # exagerated to avoid early finish
    deltares = optim_args['deltares'] if 'deltares' in optim_args else 1e-4
    rho = optim_args['rho'] if 'rho' in optim_args else 0.95
    eps = optim_args['eps'] if 'eps' in optim_args else 0.1
    alpha = optim_args['alpha'] if 'alpha' in optim_args else 0.7
    psi = min(optim_args['psi'], alpha) if 'psi' in optim_args else alpha
    return vareps, maxiter, deltares, rho, eps, alpha, psi

def projectDirection(x, d, box, eps=1e-10):
    # to avoid reaching a set of coordinates out of the constrained box
    for i in range(len(d)):
        if (abs(-box-x[i]) < eps and d[i] < 0) or (box - x[i] < eps and d[i] > 0):
            d[i] = 0 # zero out the direction dims leading out of constrained box
    return d

def solveDeflected(x, y, K, box, optim_args, target_func_value, max_error_target_func_value, return_history=True, verbose=False):
    """
        x   : initial values of betas      [ vector of zero -> linear and box constraints satisfied ]
        y   : output vector
        K   : kernel matrix
        box : value for box contraint      [ x in (-box, box) ]
    """
    vareps, maxiter, deltares, rho, eps, alpha, psi = unrollArgs(optim_args) # get all parameters needed for the algorithm
    xref = copy.deepcopy(x) # set reference point
    fref = math.inf # set reference function value
    delta = 0 # initial value for vanishing threshold parameter
    dprev = np.zeros((x.size,1)) # previous direction needed for deflection
    i = 0 # iteration count
    prevnormg = math.inf # gradient norm at previous step
    history = {'f': []} # dictionary needed for plotting after computation
    while True:
        if abs(fref - target_func_value) <= max_error_target_func_value:
            # acceptable condition reached
            if return_history:
                history['fstar'] = fref # save minimum function value
                return xref, 'acceptable', history
            return xref, 'acceptable', None
        if i > maxiter:
            # stopped condition reached
            if return_history:
                history['fstar'] = fref # save minimum function value
                return xref, 'stopped', history
            return xref, 'stopped', None
        v = (0.5 * np.dot(np.dot(np.transpose(x), K), x) 
            + vareps * np.sum(np.abs(x))
            - np.transpose(y).dot(x))[0,0] # would return a matrix otherwise
        g = K.dot(x) + vareps*np.sign(x) - y.reshape(-1,1) # reshape to transform y from horizontal to vertical array
        norm_g = np.linalg.norm(g) # get norm of descent direction gradient
        if verbose: print("i: {:4d} - v: {:4f} - fref: {:4f} - ||g||: {:4f} - delta: {:e} - ||gdiff||: {:4f} - eps: {:e}".format(i, v, fref, norm_g, delta, prevnormg-norm_g, eps))
        prevnormg = norm_g
        if norm_g < 1e-10:
            # optimal condition reached
            if return_history:
                history['fstar'] = v
                return x, 'optimal', history
            return x, 'optimal', None
        # reset delta if v is good or decrease it otherwise
        if v <= fref - delta:
            delta = deltares * max(abs(v),1)
        else:
            delta = max(delta*rho, eps*max(abs(min(v,fref)), 1))
        # update fref and xref if needed
        if v < fref:
            fref = copy.deepcopy(v)
            xref = copy.deepcopy(x)
        d = alpha*g + (1-alpha)*dprev # get deflected direction
        dproj = projectDirection(x, d, box) # constrain direction accordingly
        dprev = dproj 
        nu = psi*(v-fref+delta)/(np.linalg.norm(dproj)**2) # get stepsize following Target Value
        x = x - nu*dproj # get new point coordinates
        x = solveKP(box, 0, x, False) # project new point to follow constraints
        i += 1 # next iteration
        history['f'].append(v)