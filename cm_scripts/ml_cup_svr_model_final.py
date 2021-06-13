import numpy as np
import get_cup_dataset as dt
from SVR import SVR
import time
import matplotlib.pyplot as plt
import sys
import math

def plot_svr_predict(svr, x, y, pred, text="fig_title"):
    fig,axs = plt.subplots(2,5)
    for i in range(x.shape[1]):
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],y,color="red")
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],pred,color="blue")
    fig.suptitle(text)
    plt.show()

data, data_out = dt._get_cup('train')
test_split = 0.2
test_len = int(len(data)*test_split)
test, test_out = data[:test_len, :], data_out[:test_len, :]
test_out1, test_out2 = test_out[:, 0], test_out[:, 1]
dev_set, dev_out = data[test_len:, :], data_out[test_len:, :]
dev_out1, dev_out2 = dev_out[:, 0], dev_out[:, 1]

# Model configurations extracted from model selection in 'ml_cup_svr_model.py'
cup_model_1 = SVR('poly',{'gamma':5.1127589863994345, 'degree': 3, 'coef': 0.4675119966970296},box=1, eps=0.1) # values found with grid search results
cup_model_2 = SVR('rbf',{'gamma':0.1},box=1, eps=0.1)                                                          # values found with grid search results
beta_init = np.vstack(np.zeros(dev_set.shape[0]))
# Train over the entire dev_set
cup_model_1.fit(dev_set, dev_out1,optim_args={'eps': 0.01, 'maxiter': 3000.0}, beta_init=beta_init, verbose_optim=False)
cup_model_2.fit(dev_set, dev_out2,optim_args={'vareps': 0.1, 'maxiter': 3000.0}, beta_init=beta_init, verbose_optim=False)
pred_1 = [float(cup_model_1.predict(dev_set[i])) for i in range(dev_set.shape[0])]
pred_2 = [float(cup_model_2.predict(dev_set[i])) for i in range(dev_set.shape[0])]
print("SUM OF eps-LOSS:", cup_model_1.eps_ins_loss(pred_1) + cup_model_2.eps_ins_loss(pred_2))
mee = 0
for i in range(len(pred_1)):
    mee += math.sqrt((dev_out1[i] - pred_1[i])**2 + (dev_out2[i] - pred_2[i])**2)
mee = mee/len(pred_1)
print("DEVSET MEE:", mee)
plot_svr_predict(cup_model_1, dev_set, dev_out1, pred_1, text='1st output dim')
plot_svr_predict(cup_model_2, dev_set, dev_out2, pred_2, text='2nd output dim')

# Test the final model
testpred_1 = [float(cup_model_1.predict(test[i])) for i in range(test.shape[0])]
testpred_2 = [float(cup_model_2.predict(test[i])) for i in range(test.shape[0])]
print("SUM OF eps-LOSS:", cup_model_1.eps_ins_loss(test_out1, testpred_1) + cup_model_2.eps_ins_loss(test_out2, testpred_2))
mee = 0
for i in range(len(testpred_1)):
    mee += math.sqrt((test_out1[i] - testpred_1[i])**2 + (test_out2[i] - testpred_2[i])**2)
mee = mee/len(testpred_1)
print("TEST MEE:", mee)
plot_svr_predict(cup_model_1, test, test_out1, testpred_1, text='1st output dim')
plot_svr_predict(cup_model_2, test, test_out2, testpred_2, text='2nd output dim')