import numpy as np
import utils.get_dataset as dt
from utils.model import Model
from utils.layer import Layer
from utils.plot import Plot

from grid_search_ml_cup import GridSearch

# ----------------------------------------- MAIN ----------------------------------------- #
gs = GridSearch()
train, validation, test, train_labels, validation_labels, test_labels = dt._get_split_cup()
models = [
    [Layer(16, "tanh", _input=(10,)), 
    Layer(16, "tanh"),
    Layer(16, "tanh"),
    Layer(16, "tanh"),
    Layer(16, "tanh"), 
    Layer(2, "linear")],

    [Layer(8, "leaky_relu", _input=(10,)), 
    Layer(8, "leaky_relu"),
    Layer(8, "leaky_relu"),
    Layer(8, "leaky_relu"),
    Layer(8, "leaky_relu"), 
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
                eta=[5e-4,1e-4,5e-5,1e-5,5e-6],
                alpha=[0.8,0.9,0.99],
                batch_size=[len(train_labels)],
                epoch=[150],
                lr_decay=[1e-5],
                _lambda=[1e-3, 1e-4, 1e-5]
            )
gs._run(train, train_labels, validation, validation_labels, test, test_labels, familyofmodelsperconfiguration=1)
