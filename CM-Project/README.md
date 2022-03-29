# Computational Mathematics for Learning and Data Analysis

## Project 14
*(Model)* is a SVR-type approach of your choice (in particular, with one or more kernels of your choice).

*(Algorithm)* is a dual approach with appropriate choices of the constraints to be dualized, where the Lagrangian Dual is solved by an algorithm of the class of deflected subgradient methods.

*Reference*: [1][1] - [2][2]

#

## Files Description
In the folders *linear/poly* are reported all the notebooks with the experiments, in particular:

 - *get_best_model_minumum_50*: computes the **f_best** value for the best model
 - *test_best_five_#*: train single model
 - *gs_top5_res_err*: computes all the metrics (**convergence rate, log residual error**) from the previously trained models

The other files in the repository are the main scripts that contain the implementation of all the different procedures.
The most important ones are:
- *SVR*: class of the model
- *Deflected subgradient*: optimization algorithm
- *KP*: script to solve the convex separable knapsack problem

#

## How to run
It is possible to run the notebooks to create the models that are saved using pickle, and then run the appropriate *gs_top5_res_err* notebook to analyze the results.

**Please note**: some scripts may take up to 1.30h to run (depending on the number of iterations). 

Otherwise, it is possible to create one single instance of the model and run with the following code, from the main directory:

```python
import get_cup_dataset as dt
from SVR import SVR

# collect data from dataset
data, data_out = dt._get_cup('train')
dim = # [0-1]
data_out = data_out[:, dim]

# create SVR model
model = SVR(...)
# define optimization algorithm parameters
opt_args = {'alpha': ..., 'psi': ..., 'eps': ...,
			'rho': ..., 'deltares': ... , 'maxiter': ...}
# set parameters for 'acceptable' scenario
max_error_target_func = # ...
target_func_value = # ...
# fit the model
model.fit(data, data_out, optim_args=opt_argv,
		  optim_verbose=False, convergence_verbose=True,
		  max_error_target_func_value=max_error_target_func,
		  target_func_value=target_func_value)
# history contains f values during optimization and fstar
print(model.history.keys())
# print status at end of optimization process
# (acceptable, stopped, optimal)
print(model.status)
```
#
## Regarding the *gs_models* folder
No models are present inside the *gs_models* folder. This is due to the high space occupation of the model set we computed (~400 MB overall). If needed, we can promptly provide the models in a separate delivery.

<!-- References -->

[1]: http://pages.di.unipi.it/frangio/abstracts.html#MPC16

[2]: http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf