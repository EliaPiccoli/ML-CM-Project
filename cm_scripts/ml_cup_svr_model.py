import numpy as np
import get_cup_dataset as dt
from SVR import SVR
from svr_grid_search import Gridsearch
import time
import matplotlib.pyplot as plt
import sys

def search(x, y, test_x, test_y):
    gs = Gridsearch()
    gs.set_parameters(
        kernel=["linear", "rbf", "rbf", "poly", "poly", "sigmoid", "sigmoid", "sigmoid"],
        kparam=[{}, {"gamma":1}, {"gamma":10}, {"degree":2, "gamma":'auto'}, {"degree":4, "gamma":'auto'}, {"gamma":"scale"}, {"gamma":1}, {"gamma":10}],
        box=[0.1,1,10],
        eps=[0.1,0.5,1],
        optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':5e-4, 'maxiter':3e3}]
    )
    best_coarse_model = gs.run(
        x, y, test_x, test_y
    )

    print("BEST COARSE GRID SEARCH MODEL:",best_coarse_model)
    svr = best_coarse_model
    kernel, kparam, optiargs = gs.get_model_perturbations(best_coarse_model, 10, 6)
    print(kernel, kparam, optiargs)
    gs.set_parameters(
        kernel=kernel,
        kparam=kparam,
        optiargs=optiargs
    )
    best_fine_model = gs.run(
        x, y, test_x, test_y
    )
    print("BEST FINE GRID SEARCH MODEL:",best_fine_model)

    svr = best_fine_model
    pred = [float(svr.predict(x[i])) for i in range(x.shape[0])]
    print("LOSS:", svr.eps_ins_loss(pred))

# ----------------------------------------------------------------- #

def search_linear(x, y, test_x, test_y):
    gs = Gridsearch()
    gs.set_parameters(
        kernel=["linear"],
        kparam=[{}],
        box=[1,10],
        eps=[0.5,1],
        optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':5e-4, 'maxiter':3e3}, {'eps':1e-4, 'maxiter':3e3}]
    )
    best_coarse_model = gs.run(
        x, y, test_x, test_y
    )

    print("BEST COARSE GRID SEARCH MODEL:",best_coarse_model)
    svr = best_coarse_model
    kernel, kparam, optiargs = gs.get_model_perturbations(best_coarse_model, 1, 1)
    print(kernel, kparam, optiargs)
    gs.set_parameters(
        kernel=kernel,
        kparam=kparam,
        optiargs=optiargs
    )
    best_fine_model = gs.run(
        x, y, test_x, test_y
    )
    print("BEST FINE GRID SEARCH MODEL:",best_fine_model)

    svr = best_fine_model
    pred = [float(svr.predict(x[i])) for i in range(x.shape[0])]
    print("LOSS:", svr.eps_ins_loss(pred))

    fig,axs = plt.subplots(2,5)
    for i in range(x.shape[1]):
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],y,color="red")
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],pred,color="blue")
    fig.suptitle('Linear')

    plt.show()

# ----------------------------------------------------------------- #

def search_rbf(x, y, test_x, test_y):
    gs = Gridsearch()
    gs.set_parameters(
        # kernel=["poly", "poly", "poly", "poly", "poly", "poly", "poly", "poly"],
        # kparam=[{"degree":2, "gamma":'auto'},{"degree":2, "gamma":5},{"degree":3, "gamma":'auto'}, {"degree":3, "gamma":5}, {"degree":4, "gamma":'auto'}, {"degree":4, "gamma":5},{"degree":5, "gamma":'auto'},{"degree":5, "gamma":5}],
        # box=[0.1,1,10],
        # eps=[0.1,0.5,1],
        kernel=["rbf", "rbf", "rbf", "rbf", "rbf"],
        kparam=[{"gamma":'auto'},{"gamma":1},{"gamma":2},{"gamma":5},{"gamma":10}],
        optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':5e-4, 'maxiter':3e3}]
    )
    best_coarse_model = gs.run(
        x, y, test_x, test_y
    )

    print("BEST COARSE GRID SEARCH MODEL:",best_coarse_model)
    svr = best_coarse_model
    kernel, kparam, optiargs = gs.get_model_perturbations(best_coarse_model, 3, 3)
    print(kernel, kparam, optiargs)
    gs.set_parameters(
        kernel=kernel,
        kparam=kparam,
        optiargs=optiargs
    )
    best_fine_model = gs.run(
        x, y, test_x, test_y
    )
    print("BEST FINE GRID SEARCH MODEL:",best_fine_model)

    svr = best_fine_model
    pred = [float(svr.predict(x[i])) for i in range(x.shape[0])]
    print("LOSS:", svr.eps_ins_loss(pred))

    fig,axs = plt.subplots(2,5)
    for i in range(x.shape[1]):
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],y,color="red")
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],pred,color="blue")
    fig.suptitle('RBF')

    plt.show()

# --------------------------------------------------------------------------------------- #
    
def search_sigmoid(x, y, test_x, test_y):
    gs = Gridsearch()
    gs.set_parameters(
        # kernel=["poly", "poly", "poly", "poly", "poly", "poly", "poly", "poly"],
        # kparam=[{"degree":2, "gamma":'auto'},{"degree":2, "gamma":5},{"degree":3, "gamma":'auto'}, {"degree":3, "gamma":5}, {"degree":4, "gamma":'auto'}, {"degree":4, "gamma":5},{"degree":5, "gamma":'auto'},{"degree":5, "gamma":5}],
        # box=[0.1,1,10],
        # eps=[0.1,0.5,1],
        kernel=["sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid"],
        kparam=[{"gamma":'auto'},{"gamma":1},{"gamma":2},{"gamma":5},{"gamma":10}],
        optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':5e-4, 'maxiter':3e3}]
    )
    best_coarse_model = gs.run(
        x, y, test_x, test_y
    )

    print("BEST COARSE GRID SEARCH MODEL:",best_coarse_model)
    svr = best_coarse_model
    kernel, kparam, optiargs = gs.get_model_perturbations(best_coarse_model, 3, 3)
    print(kernel, kparam, optiargs)
    gs.set_parameters(
        kernel=kernel,
        kparam=kparam,
        optiargs=optiargs
    )
    best_fine_model = gs.run(
        x, y, test_x, test_y
    )
    print("BEST FINE GRID SEARCH MODEL:",best_fine_model)

    svr = best_fine_model
    pred = [float(svr.predict(x[i])) for i in range(x.shape[0])]
    print("LOSS:", svr.eps_ins_loss(pred))

    fig,axs = plt.subplots(2,5)
    for i in range(x.shape[1]):
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],y,color="red")
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],pred,color="blue")
    fig.suptitle('Sigmoid')

    plt.show()
# -------------------------------------------------------------------------------------------- #

def search_poly(x, y, test_x, test_y):
    gs = Gridsearch()
    gs.set_parameters(
        # kernel=["poly", "poly", "poly", "poly", "poly", "poly", "poly", "poly"],
        # kparam=[{"degree":2, "gamma":'auto'},{"degree":2, "gamma":5},{"degree":3, "gamma":'auto'}, {"degree":3, "gamma":5}, {"degree":4, "gamma":'auto'}, {"degree":4, "gamma":5},{"degree":5, "gamma":'auto'},{"degree":5, "gamma":5}],
        # box=[0.1,1,10],
        # eps=[0.1,0.5,1],
        kernel=["poly", "poly", "poly", "poly", "poly"],
        kparam=[{"degree":2, "gamma":1},{"degree":3, "gamma":1},{"degree":4, "gamma":1},{"degree":5, "gamma":1},{"degree":10, "gamma":1}],
        optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':5e-4, 'maxiter':3e3}]
    )
    best_coarse_model = gs.run(
        x, y, test_x, test_y
    )

    print("BEST COARSE GRID SEARCH MODEL:",best_coarse_model)
    svr = best_coarse_model
    kernel, kparam, optiargs = gs.get_model_perturbations(best_coarse_model, 3, 3)
    print(kernel, kparam, optiargs)
    gs.set_parameters(
        kernel=kernel,
        kparam=kparam,
        optiargs=optiargs
    )
    best_fine_model = gs.run(
        x, y, test_x, test_y
    )
    print("BEST FINE GRID SEARCH MODEL:",best_fine_model)

    svr = best_fine_model
    pred = [float(svr.predict(x[i])) for i in range(x.shape[0])]
    print("LOSS:", svr.eps_ins_loss(pred))

    fig,axs = plt.subplots(2,5)
    for i in range(x.shape[1]):
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],y,color="red")
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],pred,color="blue")
    fig.suptitle('Poly1')

    plt.show()

# -------------------------------------------------------------------------------------------- #

def search_poly2(x, y, test_x, test_y):
    gs = Gridsearch()
    gs.set_parameters(
        # kernel=["poly", "poly", "poly", "poly", "poly", "poly", "poly", "poly"],
        # kparam=[{"degree":2, "gamma":'auto'},{"degree":2, "gamma":5},{"degree":3, "gamma":'auto'}, {"degree":3, "gamma":5}, {"degree":4, "gamma":'auto'}, {"degree":4, "gamma":5},{"degree":5, "gamma":'auto'},{"degree":5, "gamma":5}],
        # box=[0.1,1,10],
        # eps=[0.1,0.5,1],
        kernel=["poly", "poly", "poly", "poly", "poly"],
        kparam=[{"degree":3, "gamma":'auto'},{"degree":3, "gamma":1},{"degree":3, "gamma":2},{"degree":3, "gamma":5},{"degree":3, "gamma":10}],
        optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':5e-4, 'maxiter':3e3}]
    )
    best_coarse_model = gs.run(
        x, y, test_x, test_y
    )

    print("BEST COARSE GRID SEARCH MODEL:",best_coarse_model)
    svr = best_coarse_model
    kernel, kparam, optiargs = gs.get_model_perturbations(best_coarse_model, 3, 3)
    print(kernel, kparam, optiargs)
    gs.set_parameters(
        kernel=kernel,
        kparam=kparam,
        optiargs=optiargs
    )
    best_fine_model = gs.run(
        x, y, test_x, test_y
    )
    print("BEST FINE GRID SEARCH MODEL:",best_fine_model)

    svr = best_fine_model
    pred = [float(svr.predict(x[i])) for i in range(x.shape[0])]
    print("LOSS:", svr.eps_ins_loss(pred))

    fig,axs = plt.subplots(2,5)
    for i in range(x.shape[1]):
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],y,color="red")
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],pred,color="blue")
    fig.suptitle('Poly2')

    plt.show()




start = time.time()

grid = True
train, train_labels = dt._get_cup('train')
test, test_labels = train[:len(train)//10], train_labels[:len(train_labels)//10]
test_labels1, test_labels2 = test_labels[:,0], test_labels[:,1]
train, train_labels = train[len(train)//10:], train_labels[len(train_labels)//10:]

# if not grid:
#     val = len(train) # for faster trials
#     train = train[:val,:]
#     train_labels = train_labels[:val,:]
#     train_labels1, train_labels2 = train_labels[:,0], train_labels[:,1]
#     cup_model_1 = SVR('rbf',{'gamma':'scale'},box=10, eps=0.5)
#     cup_model_2 = SVR('rbf',{'gamma':'scale'},box=10, eps=0.5)
#     beta_init = np.vstack(np.zeros(train.shape[0]))
#     # print("beta_init", beta_init)
#     cup_model_1.fit(train, train_labels1,optim_args={'maxiter':1e3, 'eps':1e-2}, beta_init=beta_init)
#     cup_model_2.fit(train, train_labels2,optim_args={'maxiter':1e3, 'eps':1e-2}, beta_init=beta_init)
#     pred_1 = [float(cup_model_1.predict(train[i])) for i in range(train.shape[0])]
#     pred_2 = [float(cup_model_2.predict(train[i])) for i in range(train.shape[0])]
#     print("LOSS:", cup_model_1.eps_ins_loss(pred_1) + cup_model_2.eps_ins_loss(pred_2))

print("Training data:",len(train))

if grid: 
    train_labels1, train_labels2 = train_labels[:,0], train_labels[:,1]
    if sys.argv[1] == 'linear':
        print("Linear gridsearch..")
        search_linear(train, train_labels1, test, test_labels1)
    elif sys.argv[1] == 'rbf':
        print("RBF gridsearch..")
        search_rbf(train, train_labels1, test, test_labels1)
    elif sys.argv[1] == 'poly':
        print("Poly gridsearch..")
        search_poly2(train, train_labels1, test, test_labels1)
    elif sys.argv[1] == 'sigmoid':
        print("Sigmoid gridsearch..")
        search_sigmoid(train, train_labels1, test, test_labels1)

print("Time taken:",time.time()-start)


