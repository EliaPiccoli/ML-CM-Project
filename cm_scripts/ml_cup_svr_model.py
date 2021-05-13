import numpy as np
import get_cup_dataset as dt
from SVR import SVR



train, train_labels = dt._get_cup('train')
test, test_labels = dt._get_cup('test')
val = 10
train = train[:val,:]
train_labels = train_labels[:val,:]
train_labels1, train_labels2 = train_labels[:,0], train_labels[:,1]

cup_model = SVR('rbf',{'gamma':'scale'},box=10, eps=0.5)
beta_init = np.vstack(np.zeros(train.shape[0]))
# print("beta_init", beta_init)
cup_model.fit(train, train_labels1,optim_args={'maxiter':1e2, 'eps':1e-2}, beta_init=beta_init)



