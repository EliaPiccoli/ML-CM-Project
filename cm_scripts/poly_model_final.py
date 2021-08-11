import numpy as np
import get_cup_dataset as dt
from SVR import SVR
import time
import matplotlib.pyplot as plt
import sys
import math

x = np.vstack(np.arange(-50,51,1))
degree = 2
noising_factor = 0.1
np.random.seed(0)
y = [xi**degree+np.random.normal(0,1) for xi in x]
y = np.array(y, dtype=np.float64)

test_x = [-25, -20, -18, -15, -12, 12, 15, 18, 20, 25]
test_y = [xi**degree for xi in test_x]


# Model configurations extracted from model selection in 'ml_cup_svr_model.py'
model = SVR('poly',{'gamma':1.0, 'degree':2}, box=10, eps=0.5) # values found with grid search results
beta_init = np.vstack(np.zeros(x.shape[0]))
# Train over the entire dev_set
print("Training first model ... ")
model.fit(x, y, optim_args={'eps': 0.01, 'maxiter': 3000.0, 'vareps': 0.5}, beta_init=beta_init, verbose_optim=True, convergence_verbose=True)
pred = [float(model.predict(x[i])) for i in range(x.shape[0])]
print("eps-LOSS:", model.eps_ins_loss(y, pred))
mee = 0
for i in range(len(pred)):
    mee += np.sqrt((y[i] - pred[i])**2)
mee = mee/len(pred)
print(f"DEVSET MEE: {mee}")

# Test the final model
print("Testing the model")
testpred = [float(model.predict(test_x[i])) for i in range(len(test_x))]
print("eps-LOSS:", model.eps_ins_loss(test_y, testpred))
mee = 0
for i in range(len(testpred)):
    mee += np.sqrt((test_y[i] - testpred[i])**2)
mee = mee/len(testpred)
print(f"TEST MEE: {mee}")

_,axs=plt.subplots(1,2)
axs[0].scatter(x, y , color="red")
axs[0].plot(x, pred, color="blue")
axs[0].set_title('poly SVR - dev')
axs[0].set_xlabel('Input')
axs[0].set_ylabel('Output')
axs[1].scatter(test_x, test_y , color="red")
axs[1].plot(test_x, testpred, color="blue")
axs[1].set_title('poly SVR - test')
axs[1].set_xlabel('Input')
axs[1].set_ylabel('Output')
plt.show()