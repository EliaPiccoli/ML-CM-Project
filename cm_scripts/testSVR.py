import numpy as np
from sklearn.preprocessing import StandardScaler
import kernel
from deflected_subgradient import solveDeflected
from SVR import *
import matplotlib.pyplot as plt

x = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([[45000],[50000],[60000],[80000],[110000],[150000],[200000],[300000],[500000],[1000000]])

sc_X = StandardScaler()
sc_y = StandardScaler()
x = sc_X.fit_transform(x)
y = sc_y.fit_transform(y)

# GK = kernel.rbf(x)
# box = 1.0
# x_init = np.zeros(x.shape)
# beta = solveDeflected(x_init, y, K, box, {}, True)
# print("LOL: ", beta)

K = kernel.poly(x, 'scale', 2)
box = 4.0
DEG = 3
x_init = np.zeros(x.shape)
beta = solveDeflected(x_init, y, K, box, {}, True)
print("LOL: ", beta)

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
b -= 0.1 # -eps
b /= len(beta) # (why ?) (computing average bias ??)
# for i in range(beta.size):
#     if beta[i] > 1e-10: # active point
#         b = -y[i] + np.dot(np.transpose(W), x[i]) - 0.1
#         break
print(f"W : {W} - b: {b}")

# First transform 6.5 to feature scaling
sc_X_val = sc_X.transform(np.array([[6.5]]))
# Second predict the value
# scaled_y_pred = predict_gk(b, beta, sc_X_val, suppvect)
scaled_y_pred = predict_poly(b, beta, sc_X_val, suppvect, DEG)
# Third - since this is scaled - we have to inverse transform
y_pred = sc_y.inverse_transform(scaled_y_pred) 
print('The predicted salary of a person at 6.5 Level is ', y_pred)

plt.scatter(x, y , color="red")
# pred = [float(predict_gk(b, beta, x[i], suppvect, 1)) for i in range(x.size)]
pred = [float(predict_poly(b, beta, x[i], suppvect, DEG)) for i in range(x.size)]
print(pred)
plt.plot(x, pred, color="blue")
plt.title("SVR")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()