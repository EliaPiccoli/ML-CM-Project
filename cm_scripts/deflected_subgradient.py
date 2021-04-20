from kp import solveKP
import kernel
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
    maxiter = optim_args['maxiter'] if 'maxiter' in optim_args else 10000
    deltares = optim_args['deltares'] if 'deltares' in optim_args else 1e-4
    rho = optim_args['rho'] if 'rho' in optim_args else 0.95
    eps = optim_args['eps'] if 'eps' in optim_args else 1e-1
    alpha = optim_args['alpha'] if 'alpha' in optim_args else 0.7
    psi = optim_args['psi'] if 'psi' in optim_args else 0.7
    return vareps, maxiter, deltares, rho, eps, alpha, psi

def projectDirection(x, d, box, eps=1e-10):
    for i in range(len(d)):
        if (abs(-box-x[i]) < eps and d[i] < 0) or (box - x[i] < eps and d[i] > 0):
            d[i] = 0
    return d

def solveDeflected(x, y, K, box, optim_args, verbose=False):
    """
        x   : initial values of betas      [ vector of zero -> linear and box constraints satisfied ]
        y   : output vector
        K   : kernel matrix
        box : value for box contraint      [ x in (-box, box) ]
    """
    vareps, maxiter, deltares, rho, eps, alpha, psi = unrollArgs(optim_args)
    xref = copy.deepcopy(x)
    fref = math.inf
    delta = 0
    dprev = np.zeros((x.size,1))
    i = 0
    prevnormg = math.inf
    zigzagcount = 0
    while True:
        if i > maxiter:
            # stopped condition reached
            # TODO ADD status
            return xref
        v = (0.5 * np.dot(np.dot(np.transpose(x), K), x) 
            + np.repeat(vareps,x.size).dot(np.abs(x)) 
            - np.transpose(y).dot(x))[0,0] # would return a matrix otherwise
        g = K.dot(x) + vareps*np.sign(x) - y
        norm_g = np.linalg.norm(g)
        if verbose: print("i: {:4d} - v: {:4f} - fref: {:4f} - ||g||: {:4f} - delta: {:e} - ||gdiff||: {:4f} - eps: {:e}".format(i, v, fref, norm_g, delta, prevnormg-norm_g, eps))
        if prevnormg-norm_g < -1e-4:
            zigzagcount += 1
            if zigzagcount > 100 and eps > 1e-5:
                zigzagcount = 0
                eps /= 10
        prevnormg = norm_g
        if norm_g < 1e-10:
            # optimal condition reached
            # TODO ADD status
            return x
        # reset delta if v is good or decrease it otherwise
        if v <= fref - delta:
            delta = deltares * max(abs(v),1)
        else:
            delta = max(delta*rho, eps*max(abs(min(v,fref)), 1))
        # update fref and xref if needed
        if v < fref:
            fref = copy.deepcopy(v)
            xref = copy.deepcopy(x)
        d = alpha*g + (1-alpha)*dprev
        dproj = projectDirection(x, d, box)
        dprev = dproj
        # print("dproj: ", dproj)
        nu = psi*(v-fref+delta)/(np.linalg.norm(dproj)**2)
        # print("nu: ", nu)
        # print("current: ",x)
        x = x - nu*dproj
        # print("updated: ",x) 
        x = solveKP(box, 0, x, False)
        # print("projected: ",x)
        i += 1
        # input()