import numpy as np
import utils.get_dataset as dt
from utils.model import Model
from utils.layer import Layer
from utils.plot import Plot
import random

MONK_MODEL = 1

if MONK_MODEL == 1:
<<<<<<< HEAD
    seed = 10
=======
    seed = 10
>>>>>>> d938def0979aa1021573c90d91f443679dda1d63
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
    model._add_layer(Layer(1, "sigmoid"))
    model._compile(eta=0.05, loss_function="mse", alpha=0.9, stopping_eta=1, weight_range=(-0.01, 0.01), seed=seed)
    epoch = 500
    stats = model._train(ohe_inp, train_exp, ohe_val, validation_exp, decay=5e-5,batch_size=len(ohe_inp), epoch=epoch, verbose=True)

    # testing the model
    print("Test Accuracy: {:.6f}".format(model._infer(ohe_test, test_exp)))
elif MONK_MODEL == 2:
    seed = 123
    train, validation, train_labels, validation_labels = dt._get_train_validation_data(MONK_MODEL, split=0.2, seed=seed)
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
    model._add_layer(Layer(1, "sigmoid"))
    model._compile(eta=0.04, loss_function="mse", alpha=0.9, stopping_eta=1, _lambda=0, weight_range=(-0.01, 0.01), seed=seed)
    epoch = 500
    stats = model._train(ohe_inp, train_exp, ohe_val, validation_exp, decay=12e-6, batch_size=len(ohe_inp), epoch=epoch, verbose=True)

    # testing the model
    print("Test Accuracy: {:.6f}".format(model._infer(ohe_test, test_exp)))
elif MONK_MODEL == 3: # no-reg
    seed = 1
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
    model._add_layer(Layer(4, "relu", _input=(17,)))
    model._add_layer(Layer(1, "sigmoid"))
    model._compile(eta=0.02, loss_function="mse", alpha=0.99, stopping_eta=1, _lambda=0, weight_range=(-0.5, 0.5), seed=seed)
    epoch = 500
    stats = model._train(ohe_inp, train_exp, ohe_val, validation_exp, decay=5e-6, batch_size=len(ohe_inp), epoch=epoch, verbose=True)

    # testing the model
    print("Test Accuracy: {:.6f}".format(model._infer(ohe_test, test_exp)))
elif MONK_MODEL == 4: # reg
    MONK_MODEL = 3
    seed = 1
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
    model._add_layer(Layer(4, "relu", _input=(17,)))
    model._add_layer(Layer(1, "sigmoid"))
    model._compile(eta=0.02, loss_function="mse", alpha=0.99, stopping_eta=1, _lambda=3.5e-3, weight_range=(-0.5, 0.5), seed=seed)
    epoch = 500
    stats = model._train(ohe_inp, train_exp, ohe_val, validation_exp, decay=1e-5, batch_size=len(ohe_inp), epoch=epoch, verbose=True)

    # testing the model
    print("Test Accuracy: {:.6f}".format(model._infer(ohe_test, test_exp)))
<<<<<<< HEAD
elif MONK_MODEL == 5: # decay-test
    MONK_MODEL = 2
    seed = 100
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
    model._add_layer(Layer(4, "relu", _input=(17,)))
    model._add_layer(Layer(1, "sigmoid"))
    model._compile(eta=0.05, loss_function="mse", alpha=0.99, stopping_eta=1, _lambda=0, weight_range=(-0.5, 0.5), seed=seed) # stopping_eta = 0.1 to test difference
    epoch = 500
    stats = model._train(ohe_inp, train_exp, ohe_val, validation_exp, decay=2e-5, batch_size=len(ohe_inp), epoch=epoch, verbose=True)

    # testing the model
    print("Test Accuracy: {:.6f}".format(model._infer(ohe_test, test_exp)))
Plot._plot_train_stats([stats], epochs=[epoch])
=======
Plot._plot_train_stats([stats], epochs=[epoch])
>>>>>>> d938def0979aa1021573c90d91f443679dda1d63
