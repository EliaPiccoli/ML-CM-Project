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
    alpha = optim_args['alpha'] if 'alpha' in optim_args else 0.8
    psi = optim_args['psi'] if 'psi' in optim_args else 0.75
    return vareps, maxiter, deltares, rho, eps, alpha, psi

def projectDirection(x, d, box, eps=1e-10):
    for i in range(len(d)):
        if (-box-x[i] < eps and d[i] < 0) or (box - x[i] < eps and d[i] > 0):
            d[i] = 0
    return d

def solveDeflected(x, y, K, box, optim_args, verbose=False):
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
        v = (0.5 * np.dot(np.dot(np.transpose(x), K), x) 
            + np.repeat(vareps,x.size).dot(np.abs(x)) 
            - np.transpose(y).dot(x))[0,0] # would return a matrix otherwise
        g = K.dot(x) + vareps*np.sign(x) - y
        norm_g = np.linalg.norm(g)
        if verbose: print("i: {:4d} - v: {:e} - fref: {:e} - ||g||: {:e}".format(i, v, fref, norm_g))
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
            fref = copy.deepcopy(v)
            xref = copy.deepcopy(x)
        d = alpha*g + (1-alpha)*dprev
        dproj = projectDirection(x, d, box)
        dprev = dproj
        print("d", dproj)
        nu = psi*(v-fref+delta)/(np.linalg.norm(dproj)**2)
        print("nu", nu)
        print("current: ",x)
        x = x - nu*dproj
        print("updated: ",x) 
        x = solveKP(box, 0, x, False)
        print("projected: ",x)
        i += 1
        input()
        # DEBUG:
        # 1. d ultime tre componenti sempre 0, valori sempre molto simili
        # 2. nu è sempre nell'ordine di e-5 troppo piccolo
        # 3. punto (2) implica che non si avanza di niente
        # 4. punto (2) implica che x si distacca di poco da fref -> nu è piccolo... è un cane che si morde la coda
        # 5. fref non sarà il valore di v al passo precedente? è possibile che io vado ad una x peggiore? Come? Se supero il minimo? Is strange

def predict(W, b, x):
    return np.dot(np.transpose(W), x) + b

def predict_gk(W, b, beta, x, sv):
    gamma = 1/(sv.shape[0]*sv.var())
    K = np.zeros((sv.shape[0], x.shape[0]))
    for i in range(len(K)):
        for j in range(len(K[0])):
            K[i,j] = np.exp(-gamma * np.linalg.norm(sv[i]-x[j])**2)
    # print(K)
    return np.dot(np.transpose(beta), K) + b

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
    x_init = np.zeros(x.shape)
    # print(x, x_init)
    beta = solveDeflected(x_init, y, K, box, {}, True)
    print("LOL", beta)

    # NON VA UN CAZZO
    # https://medium.com/analytics-vidhya/machine-learning-project-4-predict-salary-using-support-vector-regression-dd519e549468
    # https://github.com/dmeoli/optiml/blob/master/optiml/ml/svm/_base.py

    mask = np.logical_or(beta > 1e-6, beta < -1e-6)
    support = np.vstack(np.vstack(np.arange(len(beta)))[mask])
    suppvect = np.vstack(x[mask])
    y_sv = np.vstack(y[mask])
    beta = np.vstack(beta[mask])

    # only for linear kernel ??
    W = np.dot(np.transpose(beta), suppvect)

    # is it correct (?)
    b = 0
    for i in range(beta.size):
        b += y_sv[i]
        b -= np.sum(beta * K[support[i], np.hstack(mask)])
    b -= 0.1
    b /= len(beta) # (why ?) (computing average bias ??)
    # for i in range(beta.size):
    #     if beta[i] > 1e-10: # active point
    #         b = -y[i] + np.dot(np.transpose(W), x[i]) - 0.1
    #         break
    print(f"W : {W} - b: {b}")

    # First transform 6.5 to feature scaling
    sc_X_val = sc_X.transform(np.array([[6.5]]))
    # Second predict the value
    scaled_y_pred = predict(W, b, sc_X_val)
    # Third - since this is scaled - we have to inverse transform
    y_pred = sc_y.inverse_transform(scaled_y_pred) 
    print('The predicted salary of a person at 6.5 Level is ',y_pred)

    # import matplotlib.pyplot as plt
    # plt.scatter(x, y , color="red")
    # pred = [float(predict_gk(W, b, beta, x[i], suppvect)) for i in range(x.size)]
    # print(pred)
    # plt.plot(x, pred, color="blue")
    # plt.title("SVR")
    # plt.xlabel("Position")
    # plt.ylabel("Salary")
    # plt.show()