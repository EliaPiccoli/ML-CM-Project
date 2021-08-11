import numpy as np
import get_cup_dataset as dt
from SVR import SVR
from svr_grid_search import Gridsearch
import time
import matplotlib.pyplot as plt
import sys
import math

def search(x, y, val_x, val_y):
    gs = Gridsearch()
    gs.set_parameters(
        kernel=["linear", "rbf", "rbf", "poly", "sigmoid", "sigmoid"],
        kparam=[{}, {"gamma":1}, {"gamma":10}, {"degree":3, "gamma":'auto'}, {"gamma":"scale"}, {"gamma":1}],
        box=[0.1,1,10],
        eps=[0.05,0.1,0.5],
        optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':5e-3, 'maxiter':5e3}]
    )
    best_coarse_model = gs.run(
        x, y, val_x, val_y
    )

    print("BEST COARSE GRID SEARCH MODEL:", best_coarse_model)
    svr = best_coarse_model
    kernel, kparam, optiargs, eps, box  = gs.get_model_perturbations(best_coarse_model, 10, 6)
    print(kernel, kparam, optiargs)
    gs.set_parameters(
        eps=eps,
        box=box,
        kernel=kernel,
        kparam=kparam,
        optiargs=optiargs
    )
    best_fine_model = gs.run(
        x, y, val_x, val_y
    )
    print("BEST FINE GRID SEARCH MODEL:", best_fine_model)

    svr = best_fine_model
    pred = [float(svr.predict(x[i])) for i in range(x.shape[0])]
    print("T LOSS:", svr.eps_ins_loss(y, pred))

    pred = [float(svr.predict(val_x[i])) for i in range(val_x.shape[0])]
    print("V LOSS:", svr.eps_ins_loss(val_y, pred))

    return svr

# ----------------------------------------------------------------- #

def search_linear(x, y, val_x, val_y):
    gs = Gridsearch()
    gs.set_parameters(
        kernel=["linear"],
        kparam=[{}],
        box=[0.1,1,10],
        eps=[0.05,0.1,0.5],
        optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':5e-3, 'maxiter':5e3}, {'eps':5e-4, 'maxiter':3e3}]
    )
    best_coarse_model = gs.run(
        x, y, val_x, val_y
    )

    print("BEST COARSE GRID SEARCH MODEL:", best_coarse_model)
    svr = best_coarse_model
    kernel, kparam, optiargs, eps, box  = gs.get_model_perturbations(best_coarse_model, 1, 5, 5)
    print(kernel, kparam, optiargs)
    gs.set_parameters(
        eps=eps,
        box=box,
        kernel=kernel,
        kparam=kparam,
        optiargs=optiargs
    )
    best_fine_model = gs.run(
        x, y, val_x, val_y
    )
    print("BEST FINE GRID SEARCH MODEL:", best_fine_model)

    svr = best_fine_model
    pred = [float(svr.predict(x[i])) for i in range(x.shape[0])]
    print("T LOSS:", svr.eps_ins_loss(y, pred))

    fig,axs = plt.subplots(2,5)
    for i in range(x.shape[1]):
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],y,color="red",marker='x')
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],pred,color="blue",marker='.')
    fig.suptitle('TLinear')
    plt.show()

    pred = [float(svr.predict(val_x[i])) for i in range(val_x.shape[0])]
    print("V LOSS:", svr.eps_ins_loss(val_y, pred))

    fig,axs = plt.subplots(2,5)
    for i in range(val_x.shape[1]):
        axs[i//(val_x.shape[1]//2)][i%(val_x.shape[1]//2)].scatter(val_x[:,i],val_y,color="red",marker='x')
        axs[i//(val_x.shape[1]//2)][i%(val_x.shape[1]//2)].scatter(val_x[:,i],pred,color="blue",marker='.')
    fig.suptitle('VLinear')
    plt.show()
    
    return svr

# ----------------------------------------------------------------- #

def search_rbf(x, y, val_x, val_y):
    gs = Gridsearch()
    gs.set_parameters(
        kernel=["rbf", "rbf", "rbf", "rbf"],
        kparam=[{"gamma":'auto'},{"gamma":0.1},{"gamma":1},{"gamma":2}],
        box=[0.1,1,10],
        eps=[0.05,0.1,0.5],
        optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':5e-3, 'maxiter':5e3}, {'eps':5e-4, 'maxiter':3e3}]
    )
    best_coarse_model = gs.run(
        x, y, val_x, val_y
    )

    print("BEST COARSE GRID SEARCH MODEL:", best_coarse_model)
    svr = best_coarse_model
    kernel, kparam, optiargs, eps, box  = gs.get_model_perturbations(best_coarse_model, 3, 3, 3)
    print(kernel, kparam, optiargs)
    gs.set_parameters(
        eps=eps,
        box=box,
        kernel=kernel,
        kparam=kparam,
        optiargs=optiargs
    )
    best_fine_model = gs.run(
        x, y, val_x, val_y
    )
    print("BEST FINE GRID SEARCH MODEL:", best_fine_model)

    svr = best_fine_model
    pred = [float(svr.predict(x[i])) for i in range(x.shape[0])]
    print("T LOSS:", svr.eps_ins_loss(y, pred))

    fig,axs = plt.subplots(2,5)
    for i in range(x.shape[1]):
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],y,color="red",marker='x')
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],pred,color="blue",marker='.')
    fig.suptitle('TRBF')
    plt.show()

    pred = [float(svr.predict(val_x[i])) for i in range(val_x.shape[0])]
    print("V LOSS:", svr.eps_ins_loss(val_y, pred))

    fig,axs = plt.subplots(2,5)
    for i in range(val_x.shape[1]):
        axs[i//(val_x.shape[1]//2)][i%(val_x.shape[1]//2)].scatter(val_x[:,i],val_y,color="red",marker='x')
        axs[i//(val_x.shape[1]//2)][i%(val_x.shape[1]//2)].scatter(val_x[:,i],pred,color="blue",marker='.')
    fig.suptitle('VRBF')
    plt.show()
    
    return svr

# --------------------------------------------------------------------------------------- #
    
def search_sigmoid(x, y, val_x, val_y):
    gs = Gridsearch()
    gs.set_parameters(
        kernel=["sigmoid", "sigmoid", "sigmoid", "sigmoid"],
        kparam=[{"gamma":'auto'},{"gamma":1},{"gamma":2},{"gamma":5}],
        box=[0.1,1,10],
        eps=[0.05,0.1,0.5],
        optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':5e-3, 'maxiter':5e3}, {'eps':5e-4, 'maxiter':3e3}]
    )
    best_coarse_model = gs.run(
        x, y, val_x, val_y
    )

    print("BEST COARSE GRID SEARCH MODEL:", best_coarse_model)
    svr = best_coarse_model
    kernel, kparam, optiargs, eps, box  = gs.get_model_perturbations(best_coarse_model, 3, 3, 3)
    print(kernel, kparam, optiargs)
    gs.set_parameters(
        eps=eps,
        box=box,
        kernel=kernel,
        kparam=kparam,
        optiargs=optiargs
    )
    best_fine_model = gs.run(
        x, y, val_x, val_y
    )
    print("BEST FINE GRID SEARCH MODEL:", best_fine_model)

    svr = best_fine_model
    pred = [float(svr.predict(x[i])) for i in range(x.shape[0])]
    print("T LOSS:", svr.eps_ins_loss(y, pred))

    fig,axs = plt.subplots(2,5)
    for i in range(x.shape[1]):
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],y,color="red",marker='x')
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],pred,color="blue",marker='.')
    fig.suptitle('TSigmoid')
    plt.show()

    pred = [float(svr.predict(val_x[i])) for i in range(val_x.shape[0])]
    print("V LOSS:", svr.eps_ins_loss(val_y, pred))

    fig,axs = plt.subplots(2,5)
    for i in range(val_x.shape[1]):
        axs[i//(val_x.shape[1]//2)][i%(val_x.shape[1]//2)].scatter(val_x[:,i],val_y,color="red",marker='x')
        axs[i//(val_x.shape[1]//2)][i%(val_x.shape[1]//2)].scatter(val_x[:,i],pred,color="blue",marker='.')
    fig.suptitle('VSigmoid')
    plt.show()
    
    return svr

# -------------------------------------------------------------------------------------------- #

def search_poly(x, y, val_x, val_y):
    gs = Gridsearch()
    gs.set_parameters(
        kernel=["poly", "poly", "poly", "poly"],
        kparam=[{"degree":2, "gamma":1},{"degree":3, "gamma":1},{"degree":4, "gamma":1},{"degree":5, "gamma":1}],
        box=[0.1,1,10],
        eps=[0.05,0.1,0.5],
        optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':5e-3, 'maxiter':5e3}, {'eps':5e-4, 'maxiter':3e3}]
    )
    best_coarse_model = gs.run(
        x, y, val_x, val_y
    )

    print("BEST COARSE GRID SEARCH MODEL:",best_coarse_model)
    svr = best_coarse_model
    kernel, kparam, optiargs, eps, box  = gs.get_model_perturbations(best_coarse_model, 3, 3, 3)
    print(kernel, kparam, optiargs)
    gs.set_parameters(
        eps=eps,
        box=box,
        kernel=kernel,
        kparam=kparam,
        optiargs=optiargs
    )
    best_fine_model = gs.run(
        x, y, val_x, val_y
    )
    print("BEST FINE GRID SEARCH MODEL:",best_fine_model)

    svr = best_fine_model
    pred = [float(svr.predict(x[i])) for i in range(x.shape[0])]
    print("T LOSS:", svr.eps_ins_loss(y, pred))

    fig,axs = plt.subplots(2,5)
    for i in range(x.shape[1]):
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],y,color="red",marker='x')
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],pred,color="blue",marker='.')
    fig.suptitle('TPoly1')
    plt.show()

    pred = [float(svr.predict(val_x[i])) for i in range(val_x.shape[0])]
    print("V LOSS:", svr.eps_ins_loss(val_y, pred))

    fig,axs = plt.subplots(2,5)
    for i in range(val_x.shape[1]):
        axs[i//(val_x.shape[1]//2)][i%(val_x.shape[1]//2)].scatter(val_x[:,i],val_y,color="red",marker='x')
        axs[i//(val_x.shape[1]//2)][i%(val_x.shape[1]//2)].scatter(val_x[:,i],pred,color="blue",marker='.')
    fig.suptitle('VPoly1')
    plt.show()
    
    return svr

# -------------------------------------------------------------------------------------------- #

def search_polydeg3(x, y, val_x, val_y):
    gs = Gridsearch()
    gs.set_parameters(
        kernel=["poly", "poly", "poly", "poly"],
        kparam=[{"degree":3, "gamma":'auto'},{"degree":3, "gamma":1},{"degree":3, "gamma":2},{"degree":3, "gamma":5}],
        box=[0.1,1,10],
        eps=[0.05,0.1,0.5],
        optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':5e-3, 'maxiter':5e3}, {'eps':5e-4, 'maxiter':3e3}]
    )
    best_coarse_model = gs.run(
        x, y, val_x, val_y
    )

    print("BEST COARSE GRID SEARCH MODEL:", best_coarse_model)
    svr = best_coarse_model
    kernel, kparam, optiargs, eps, box = gs.get_model_perturbations(best_coarse_model, 3, 3, 3)
    print(kernel, kparam, optiargs)
    gs.set_parameters(
        eps=eps,
        box=box,
        kernel=kernel,
        kparam=kparam,
        optiargs=optiargs
    )
    best_fine_model = gs.run(
        x, y, val_x, val_y
    )
    print("BEST FINE GRID SEARCH MODEL:", best_fine_model)

    svr = best_fine_model
    pred = [float(svr.predict(x[i])) for i in range(x.shape[0])]
    print("T LOSS:", svr.eps_ins_loss(y, pred))

    fig,axs = plt.subplots(2,5)
    for i in range(x.shape[1]):
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],y,color="red",marker='x')
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],pred,color="blue",marker='.')
    fig.suptitle('TPolyD3')
    plt.show()

    pred = [float(svr.predict(val_x[i])) for i in range(val_x.shape[0])]
    print("V LOSS:", svr.eps_ins_loss(val_y, pred))

    fig,axs = plt.subplots(2,5)
    for i in range(val_x.shape[1]):
        axs[i//(val_x.shape[1]//2)][i%(val_x.shape[1]//2)].scatter(val_x[:,i],val_y,color="red",marker='x')
        axs[i//(val_x.shape[1]//2)][i%(val_x.shape[1]//2)].scatter(val_x[:,i],pred,color="blue",marker='.')
    fig.suptitle('VPolyD3')
    plt.show()
    
    return svr

# MAIN
start = time.time()
first_dim = True
data, data_out = dt._get_cup('train')
test_split = 0.2
val_split = 0.2

test_len = int(len(data)*test_split)
test, test_out = data[:test_len, :], data_out[:test_len, :]
test_out1, test_out2 = test_out[:, 0], test_out[:, 1]
dev_set, dev_out = data[test_len:, :], data_out[test_len:, :]

val_len = int(len(dev_set)*val_split)
val, val_out = dev_set[:val_len, :], dev_out[:val_len, :]
val_out1, val_out2 = val_out[:, 0], val_out[:, 1] 
train, train_out = dev_set[val_len:, :], dev_out[val_len:, :]
train_out1, train_out2 = train_out[:, 0], train_out[:, 1]

# Training & Model selection
if first_dim: 
    print("GridSearching first y dim..")
    if sys.argv[1] == 'linear':
        print("Linear gridsearch..")
        model = search_linear(train, train_out1, val, val_out1)
    elif sys.argv[1] == 'rbf':
        print("RBF gridsearch..")
        model = search_rbf(train, train_out1, val, val_out1)
    elif sys.argv[1] == 'poly':
        print("Poly gridsearch..")
        model = search_poly(train, train_out1, val, val_out1)
    elif sys.argv[1] == 'poly3':
        print("Poly3 gridsearch..")
        model = search_polydeg3(train, train_out1, val, val_out1)
    elif sys.argv[1] == 'sigmoid':
        print("Sigmoid gridsearch..")
        model = search_sigmoid(train, train_out1, val, val_out1)
    if sys.argv[1] == 'all':
        print("All gridsearch..")
        model = search(train, train_out1, val, val_out1)
else:
    print("GridSearching second y dim..")
    if sys.argv[1] == 'linear':
        print("Linear gridsearch..")
        model = search_linear(train, train_out2, val, val_out2)
    elif sys.argv[1] == 'rbf':
        print("RBF gridsearch..")
        model = search_rbf(train, train_out2, val, val_out2)
    elif sys.argv[1] == 'poly':
        print("Poly gridsearch..")
        model = search_poly(train, train_out2, val, val_out2)
    elif sys.argv[1] == 'sigmoid':
        print("Sigmoid gridsearch..")
        model = search_sigmoid(train, train_out2, val, val_out2)

print("Time taken:", time.time()-start)

# Testing the model
model_pred = []
error = 0
for inp in test:
    prediction = model.predict(inp)
    model_pred.append(prediction)
for j, val_pred in enumerate(model_pred):
    gt = test_out1[j] if first_dim else test_out2[j]
    error += math.sqrt((gt - float(val_pred))**2)
    # print(gt, val_pred)
error = error/len(model_pred)
pred = [float(model.predict(test[i])) for i in range(test.shape[0])]
print("LOSS:", model.eps_ins_loss(test_out1 if first_dim else test_out2, pred), " - MEE", error)