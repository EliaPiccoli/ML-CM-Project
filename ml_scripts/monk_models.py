import numpy as np
import utils.get_dataset as dt
from utils.model import Model
from utils.layer import Layer
from utils.plot import Plot

# ----------------------------------------- MAIN ----------------------------------------- #
print("One day I will be a very smart Artificial Intelligence!")
MONK_MODEL = 1

if MONK_MODEL == 1:
    seed = 1234
    train, validation, train_labels, validation_labels = dt._get_train_validation_data(MONK_MODEL, split=0.25, seed=seed)
    ohe_inp = [dt._get_one_hot_encoding(i) for i in train]
    ohe_val = [dt._get_one_hot_encoding(i) for i in validation]
    train_exp = [[elem] for elem in train_labels]
    validation_exp = [[elem] for elem in validation_labels]
    test, test_labels = dt._get_test_data(MONK_MODEL)
    ohe_test = [dt._get_one_hot_encoding(i) for i in test]
    test_exp = [[elem] for elem in test_labels]


    # create model
    model = Model()
    model._add_layer(Layer(4, "tanh", _input=(17,)))
    model._add_layer(Layer(1, "tanh"))
    model._compile(eta=0.01, loss_function="mse", alpha=0.99, stopping_eta=0.3, weight_range=(-0.1, 0.1), seed=seed)
    epoch = 700
    stats = model._train(ohe_inp, train_exp, ohe_val, validation_exp, decay=4e-6, batch_size=len(ohe_inp), epoch=epoch, verbose=True)

    # testing the model
    print("Test Accuracy: {:.6f}".format(model._infer(ohe_test, test_exp)))


    Plot._plot_train_stats([stats], epochs=[epoch])
elif MONK_MODEL == 2:
    seed = 123
    train, validation, train_labels, validation_labels = dt._get_train_validation_data(MONK_MODEL, split=0.25, seed=seed)
    ohe_inp = [dt._get_one_hot_encoding(i) for i in train]
    ohe_val = [dt._get_one_hot_encoding(i) for i in validation]
    train_exp = [[elem] for elem in train_labels]
    validation_exp = [[elem] for elem in validation_labels]
    test, test_labels = dt._get_test_data(MONK_MODEL)
    ohe_test = [dt._get_one_hot_encoding(i) for i in test]
    test_exp = [[elem] for elem in test_labels]


    # create model
    model = Model()
    model._add_layer(Layer(4, "tanh", _input=(17,)))
    model._add_layer(Layer(1, "tanh"))
    model._compile(eta=0.015, loss_function="mse", alpha=0.9, stopping_eta=0.25, _lambda=0, weight_range=(-0.1, 0.1), seed=seed)
    epoch = 600
    stats = model._train(ohe_inp, train_exp, ohe_val, validation_exp, decay=12e-6, batch_size=len(ohe_inp), epoch=epoch, verbose=True)

    # testing the model
    print("Test Accuracy: {:.6f}".format(model._infer(ohe_test, test_exp)))


    Plot._plot_train_stats([stats], epochs=[epoch])
elif MONK_MODEL == 3: # no-reg
    seed = 123
    train, validation, train_labels, validation_labels = dt._get_train_validation_data(MONK_MODEL, split=0.25, seed=seed)
    ohe_inp = [dt._get_one_hot_encoding(i) for i in train]
    ohe_val = [dt._get_one_hot_encoding(i) for i in validation]
    train_exp = [[elem] for elem in train_labels]
    validation_exp = [[elem] for elem in validation_labels]
    test, test_labels = dt._get_test_data(MONK_MODEL)
    ohe_test = [dt._get_one_hot_encoding(i) for i in test]
    test_exp = [[elem] for elem in test_labels]


    # create model
    model = Model()
    model._add_layer(Layer(4, "tanh", _input=(17,)))
    model._add_layer(Layer(1, "tanh"))
    model._compile(eta=0.01, loss_function="mse", alpha=0.9, stopping_eta=0.25, _lambda=0, weight_range=(-0.1, 0.1), seed=seed)
    epoch = 600
    stats = model._train(ohe_inp, train_exp, ohe_val, validation_exp, decay=5e-6, batch_size=len(ohe_inp), epoch=epoch, verbose=True)

    # testing the model
    print("Test Accuracy: {:.6f}".format(model._infer(ohe_test, test_exp)))


    Plot._plot_train_stats([stats], epochs=[epoch])
elif MONK_MODEL == 4: # reg
    MONK_MODEL = 3
    seed = 123
    train, validation, train_labels, validation_labels = dt._get_train_validation_data(MONK_MODEL, split=0.25, seed=seed)
    ohe_inp = [dt._get_one_hot_encoding(i) for i in train]
    ohe_val = [dt._get_one_hot_encoding(i) for i in validation]
    train_exp = [[elem] for elem in train_labels]
    validation_exp = [[elem] for elem in validation_labels]
    test, test_labels = dt._get_test_data(MONK_MODEL)
    ohe_test = [dt._get_one_hot_encoding(i) for i in test]
    test_exp = [[elem] for elem in test_labels]


    # create model
    model = Model()
    model._add_layer(Layer(4, "tanh", _input=(17,)))
    model._add_layer(Layer(1, "tanh"))
    model._compile(eta=0.01, loss_function="mse", alpha=0.9, stopping_eta=0.25, _lambda=2e-3, weight_range=(-0.1, 0.1), seed=seed)
    epoch = 600
    stats = model._train(ohe_inp, train_exp, ohe_val, validation_exp, decay=5e-6, batch_size=len(ohe_inp), epoch=epoch, verbose=True)

    # testing the model
    print("Test Accuracy: {:.6f}".format(model._infer(ohe_test, test_exp)))


    Plot._plot_train_stats([stats], epochs=[epoch])
