import numpy as np
import os
import utils.get_dataset as dt
from utils.model import Model
from utils.layer import Layer
from utils.plot import Plot
import pickle

from grid_search_ml_cup import GridSearch

def ensemble_exec(test, test_labels, num_models=5, verbose=False):
    dir_ensemble = "models/ensemble_models"
    ensemble_models = []
    num_models = min(len(os.listdir(dir_ensemble)), num_models)
    for i in range(num_models):
        data = {}
        with open(f"models/ensemble_models/cup_ensemble{i}", 'rb') as f:
            data = pickle.load(f)
        ensemble_models.append(data['model'])
    
    avg_inference = 0.0

    for i in range(len(ensemble_models)):
        model_score = ensemble_models[i]._infer(test, test_labels)
        avg_inference += model_score
        if verbose: print(f"Model {i} result: {model_score}")
    
    return avg_inference / len(ensemble_models)

gs = GridSearch()
train, validation, test, train_labels, validation_labels, test_labels = dt._get_split_cup()
models = [
    [Layer(8, "leaky_relu", _input=(10,)), 
    Layer(8, "leaky_relu"),
    Layer(8, "leaky_relu"),
    Layer(8, "leaky_relu"), 
    Layer(2, "linear")],

    [Layer(20, "leaky_relu", _input=(10,)), 
    Layer(20, "leaky_relu"),
    Layer(20, "leaky_relu"),
    Layer(20, "leaky_relu"), 
    Layer(2, "linear")],

    [Layer(16, "leaky_relu", _input=(10,)), 
    Layer(16, "leaky_relu"),
    Layer(16, "leaky_relu"),
    Layer(2, "linear")],

    [Layer(32, "leaky_relu", _input=(10,)), 
    Layer(32, "leaky_relu"),
    Layer(32, "leaky_relu"),
    Layer(2, "linear")],

    [Layer(16, "leaky_relu", _input=(10,)), 
    Layer(16, "leaky_relu"),
    Layer(16, "leaky_relu"),
    Layer(16, "leaky_relu"),
    Layer(16, "leaky_relu"),
    Layer(2, "linear")]
]

gs._set_parameters(layers=models, 
                weight_range=[(-0.69, 0.69)],
                eta=[1e-5, 9e-6, 5e-6, 1e-6],
                alpha=[0.1,0.8,0.9],
                batch_size=[len(train_labels)],
                epoch=[200],
                lr_decay=[5e-6, 1e-6],
                _lambda=[1e-3, 1e-4]
            )
best_model, model_conf, model_infos, model_architecture = gs._run(train, train_labels, validation, validation_labels)
print("Best model configuration: ", model_conf)

layers, weight_range, batch_size, epoch, lr_decay, eta, alpha, _lambda = gs.get_model_perturbations(model_conf, model_architecture)
print("Created perturbations: [", *model_architecture, "]", weight_range, batch_size, epoch, lr_decay, eta, alpha, _lambda)
gs._set_parameters(layers=layers, 
            weight_range=weight_range,
            eta=eta,
            alpha=alpha,
            batch_size=batch_size,
            epoch=epoch,
            lr_decay=lr_decay,
            _lambda=_lambda
        )
best_model, model_conf, model_infos, model_architecture = gs._run(train, train_labels, validation, validation_labels)
print("Best model configuration: ", model_conf)

print("Best model test accuracy: {:.6f}".format(best_model._infer(test, test_labels)))
Plot._plot_train_stats([model_infos], epochs=[model_conf['epoch']])

data = {}
with open(f"models/ensemble_models/cup_ensemble0", 'rb') as f:
    data = pickle.load(f)
model = data['model']

print("Best model test accuracy: {:.6f}".format(model._infer(test, test_labels)))

print("Ensemble result is:", ensemble_exec(test, test_labels, verbose=False))