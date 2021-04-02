from kp import solveKP
import kernel
import numpy as np
import copy
import math

def unrollArgs(optim_args):
    vareps = optim_args['vareps'] if 'vareps' in optim_args else 0.1
    maxiter = optim_args['maxiter'] if 'maxiter' in optim_args else 1e3
    deltares = optim_args['deltares'] if 'deltares' in optim_args else 1e-4
    rho = optim_args['rho'] if 'rho' in optim_args else 0.95
    eps = optim_args['eps'] if 'eps' in optim_args else 1e-6
    alpha = optim_args['alpha'] if 'alpha' in optim_args else 0.75
    psi = optim_args['psi'] if 'psi' in optim_args else 0.6
    return vareps, maxiter, deltares, rho, eps, alpha, psi

def projectDirection(x, d, box, eps=1e-10):
    for i in range(len(d)):
        if (-box-x[i] < eps and d[i] < 0) or (box - x[i] < eps and d[i] > 0):
            d[i] = 0
    return d

def solveDeflected(x, y, K, box, optim_args):
    # optim_args - vareps, rho, eps, psi, alpha
    vareps, maxiter, deltares, rho, eps, alpha, psi = unrollArgs(optim_args)
    xref = copy.deepcopy(x)
    fref = math.inf
    delta = 0
    dprev = np.zeros((x.size,1))
    i = 0
    while True:
        if i > maxiter:
            # stopped condition reached
            # TODO ADD status fam
            return xref
        v = (0.5 * np.transpose(x).dot(K).dot(x) 
            + np.repeat(vareps,x.size).dot(np.abs(x)) 
            - np.transpose(y).dot(x))[0,0] # would return a matrix otherwise
        g = K.dot(x) + vareps*np.sign(x) - y
        norm_g = np.linalg.norm(g)
        print("i: {:4d} - v: {:.2f} - ||g||: {:e}".format(i, v, norm_g))
        if norm_g < 1e-10:
            # optimal condition reached
            # TODO ADD status bro
            return x
        # reset delta if v is good or decrease it otherwise
        if v <= fref - delta:
            delta = deltares * max(v,1)
        else:
            delta = max(delta*rho, eps*max(abs(min(v,fref)), 1))
        # update fref and xref if needed
        if v < fref:
            fref = v
            xref = x
        d = alpha*g + (1-alpha)*dprev
        dproj = projectDirection(x, d, box)
        dprev = dproj
        nu = psi*(v-fref+delta)/(np.linalg.norm(dproj)**2)
        x = x - nu*dproj
        # print("before",x)
        x = solveKP(box, 0, x, False)
        # print("after",x) 
        i += 1    

if __name__ == "__main__":
    # sizes = 1000
    # x = np.random.uniform(-1,1, (sizes,10))
    # K = np.identity(sizes)
    # y = np.random.uniform(-1,1, (sizes,1))
    # optim_args = {}
    # optim_args['vareps'] = 1

    from sklearn.preprocessing import StandardScaler
    x = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
    y = np.array([[45000],[50000],[60000],[80000],[110000],[150000],[200000],[300000],[500000],[1000000]])

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    x = sc_X.fit_transform(x)
    y = sc_y.fit_transform(y)
    
    K = kernel.rbf(x)
    box = 1.0
    x = solveDeflected(x, y, K, box, {})
    # print("LOL", x)