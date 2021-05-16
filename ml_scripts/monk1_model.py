import numpy as np
import utils.get_dataset as dt
from utils.model import Model
from utils.layer import Layer
from utils.plot import Plot
from grid_search import GridSearch

# ----------------------------------------- MAIN ----------------------------------------- #
print("One day I will be a very smart Artificial Intelligence!")
print("Dataset: MONK 1")
MONK_MODEL = 1

print("Initializing datasets")
# get train, validation datasets
train, validation, train_labels, validation_labels = dt._get_train_validation_data(MONK_MODEL, split=0.25)
ohe_inp = [dt._get_one_hot_encoding(i) for i in train]
ohe_val = [dt._get_one_hot_encoding(i) for i in validation]
train_exp = [[elem] for elem in train_labels]
validation_exp = [[elem] for elem in validation_labels]
# get test dataset
test, test_labels = dt._get_test_data(MONK_MODEL)
ohe_test = [dt._get_one_hot_encoding(i) for i in test]
test_exp = [[elem] for elem in test_labels]

# define data for gridsearch
print("Starting GridSearch")
gs = GridSearch()
models = [
        [Layer(4, "tanh", _input=(17,)), Layer(1, "tanh")]
        # [Layer(5, "tanh", _input=(17,)), Layer(1, "tanh")],
        # [Layer(7, "tanh", _input=(17,)), Layer(1, "tanh")]
    ]
gs._set_parameters(layers=models, 
                weight_range=[(-0.05, 0.05)],
                eta=[1e-3, 99e-4],
                alpha=[0.85, 0.9, 0.98],
                batch_size=[len(train_labels)],
                epoch=[500],
                lr_decay=[1e-5, 5e-6, 1e-6]
            )
best_model, model_conf, model_infos, model_architecture = gs._run(ohe_inp, train_exp, ohe_val, validation_exp, familyofmodelsperconfiguration=1, plot_results=True)
print("Best model configuration: ", model_conf)

# testing the model
print("Best model test accuracy: {:.6f}".format(best_model._infer(ohe_test, test_exp)))
Plot._plot_train_stats([model_infos], epochs=[model_conf['epoch']])