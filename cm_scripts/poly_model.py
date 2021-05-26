import numpy as np
from svr_grid_search import Gridsearch
import matplotlib.pyplot as plt
import time
import sys

start = time.time()


x = np.vstack(np.arange(-50,51,1))
degree = int(sys.argv[1]) if len(sys.argv) > 1 else 2
print(f"Chosen degree is: {degree}")
y = [xi**degree for xi in x]
y = np.array(y, dtype=np.float64)

test_x = [-25, -20, -18, -15, -12, 12, 15, 18, 20, 25]
test_y = [xi**degree for xi in test_x]

gs = Gridsearch()
gs.set_parameters(
    kernel=["poly", "poly", "poly", "poly", "poly"],
    kparam=[{"degree":2, "gamma":1},{"degree":3, "gamma":1},{"degree":4, "gamma":1},{"degree":5, "gamma":1},{"degree":10, "gamma":1}],
    optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':5e-4, 'maxiter':3e3}]
    # kernel=["poly", "poly", "poly", "poly", "poly", "poly"],
    # kparam=[{"degree":2, "gamma":0.2}, {"degree":2, "gamma":0.4}, {"degree":2, "gamma":0.6}, {"degree":2, "gamma":0.8}, {"degree":2, "gamma":"scale"}, {"degree":2, "gamma":"auto"}],
    # optiargs=[{'eps':1e-2, 'maxiter':3e3}, {'eps':1e-3, 'maxiter':3e3}, {'eps':3e-4, 'maxiter':3e3}]
)
best_coarse_model = gs.run(
    x, y, test_x, test_y
)

print("BEST COARSE GRID SEARCH MODEL:",best_coarse_model)

kernel, kparam, optiargs = gs.get_model_perturbations(best_coarse_model, 6, 6)
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
to_predict = 12
pred = svr.predict(to_predict)
print(f'PREDICTION (INPUT = {to_predict})', pred)
pred = [float(svr.predict(np.array([x[i]]))) for i in range(x.size)]
print("LOSS:", svr.eps_ins_loss(pred))


print("Time taken:",time.time()-start)

plt.scatter(x, y , color="red")
plt.plot(x, pred, color="blue")
plt.title(f'SVR - poly search with degree {degree}')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()