import numpy as np
import get_cup_dataset as dt
from SVR import SVR
import time
import matplotlib.pyplot as plt
import sys

def plot_svr_predict(svr, x, y, pred, text="fig_title"):
    fig,axs = plt.subplots(2,5)
    for i in range(x.shape[1]):
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],y,color="red")
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],pred,color="blue")
    fig.suptitle(text)
    plt.show()

# ------------------------------------- MAIN ------------------------------------- #

train, train_labels = dt._get_cup('train')
test, test_labels = train[:len(train)//10], train_labels[:len(train_labels)//10]
test_labels1, test_labels2 = test_labels[:,0], test_labels[:,1]
train, train_labels = train[len(train)//10:], train_labels[len(train_labels)//10:]
train_labels1, train_labels2 = train_labels[:,0], train_labels[:,1]

cup_model_1 = SVR('poly',{'gamma':5.1127589863994345, 'degree': 3, 'coef': 0.4675119966970296},box=1, eps=0.1) # values found with grid search results
cup_model_2 = SVR('rbf',{'gamma':0.1},box=1, eps=0.1)                                                          # values found with grid search results
beta_init = np.vstack(np.zeros(train.shape[0]))
cup_model_1.fit(train, train_labels1,optim_args={'eps': 0.01, 'maxiter': 3000.0}, beta_init=beta_init, verbose_optim=False)
cup_model_2.fit(train, train_labels2,optim_args={'vareps': 0.1, 'maxiter': 3000.0}, beta_init=beta_init, verbose_optim=False)
pred_1 = [float(cup_model_1.predict(train[i])) for i in range(train.shape[0])]
pred_2 = [float(cup_model_2.predict(train[i])) for i in range(train.shape[0])]
print("LOSS:", cup_model_1.eps_ins_loss(pred_1) + cup_model_2.eps_ins_loss(pred_2))

plot_svr_predict(cup_model_1, train, train_labels1, pred_1, text='1st output dim')
plot_svr_predict(cup_model_2, train, train_labels2, pred_2, text='2nd output dim')