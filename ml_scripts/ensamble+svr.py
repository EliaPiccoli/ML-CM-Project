import os
import sys
import pickle
import numpy as np
from utils.plot import Plot
import utils.get_dataset as dt
from utils.model import Model
sys.path.append(os.path.abspath('../cm_scripts'))
from SVR import SVR
import get_cup_dataset as dt1

def get_svr_model():
    train, train_labels = dt1._get_cup('train')
    train_labels1, train_labels2 = train_labels[:,0], train_labels[:,1]

    cup_model_1 = SVR('poly',{'gamma':5.1127589863994345, 'degree': 3, 'coef': 0.4675119966970296},box=1, eps=0.1) # values found with grid search results
    cup_model_2 = SVR('rbf',{'gamma':0.1},box=1, eps=0.1)                                                          # values found with grid search results
    beta_init = np.vstack(np.zeros(train.shape[0]))
    print("Fitting first dimension model..")
    cup_model_1.fit(train, train_labels1,optim_args={'eps': 0.01, 'maxiter': 3000.0}, beta_init=beta_init, verbose_optim=False)
    print("Fitting second dimension model..")
    cup_model_2.fit(train, train_labels2,optim_args={'vareps': 0.1, 'maxiter': 3000.0}, beta_init=beta_init, verbose_optim=False)
    return [cup_model_1, cup_model_2]

def ensemblewithSVR_inference(svr_model, plot=False):
    print("Inference..")

    # loading ensemble
    dir_ensemble = "models/ensemble_models"
    ensemble_models = []
    ensemble_stats = []
    num_models = len(os.listdir(dir_ensemble))
    for i in range(num_models):
        data = {}
        with open(f"models/ensemble_models/cup_ensemble{i}", 'rb') as f:
            data = pickle.load(f)
        ensemble_models.append(data['model'])
        ensemble_stats.append(data['stats'])
    if plot: Plot._plot_train_stats(ensemble_stats[:3], epochs=[len(stats) for stats in ensemble_stats[:3]], classification=False, title=f"Ensemble Models", max_graphs_per_row=3)
    if plot: Plot._plot_train_stats(ensemble_stats[3:], epochs=[len(stats) for stats in ensemble_stats[3:]], classification=False, title=f"Ensemble Models", max_graphs_per_row=2)

    test, _ = dt._get_cup('test')
    out = []
    for inp in test:
        output = np.array([0, 0])
        # compute blind test ensemble
        for i in range(num_models):
            output = output + np.array(ensemble_models[i]._feed_forward(inp))
        ens_out = output/num_models
        # compute blind test svr
        svr_out = np.array([float(svr_model[0].predict(inp)), float(svr_model[1].predict(inp))])
        out.append(0.5*ens_out+0.5*svr_out)

    # save results
    with open("ff15_ML-CUP-TS.csv", "w") as f:
        f.write("# Elia Piccoli, Nicola Gugole\n")
        f.write("# ff15\n")
        f.write("# ML-CUP20 v1\n")
        f.write("# 15-06-2021\n")
        for i,o in enumerate(out):
            f.write(f"{i+1},{o[0]},{o[1]}\n")

    return out

    
if __name__=="__main__":
    svr_model = get_svr_model()
    ensemblewithSVR_inference(svr_model)