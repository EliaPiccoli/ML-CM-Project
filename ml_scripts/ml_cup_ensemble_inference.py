import os
import pickle
import numpy as np
from utils.plot import Plot
import utils.get_dataset as dt
from utils.model import Model

def ensemble_inference(test_file, plot=False):
    dir_ensemble = "models/ensemble_models"
    ensemble_models = []
    ensemble_stats = []
    num_models = len(os.listdir(dir_ensemble))
    # load models and plot
    for i in range(num_models):
        data = {}
        with open(f"models/ensemble_models/cup_ensemble{i}", 'rb') as f:
            data = pickle.load(f)
        ensemble_models.append(data['model'])
        ensemble_stats.append(data['stats'])
    if plot: Plot._plot_train_stats(ensemble_stats[:3], epochs=[len(stats) for stats in ensemble_stats[:3]], classification=False, title=f"Ensemble Models", max_graphs_per_row=3)
    if plot: Plot._plot_train_stats(ensemble_stats[3:], epochs=[len(stats) for stats in ensemble_stats[3:]], classification=False, title=f"Ensemble Models", max_graphs_per_row=2)

    # compute blind test
    test, _ = dt._get_cup('test')
    out = []
    for inp in test:
        output = np.array([0, 0])
        for i in range(num_models):
            output = output + np.array(ensemble_models[i]._feed_forward(inp))
        out.append(output/num_models)
    # save results
    with open("ff15_ML-CUP-TS.csv", "w") as f:
        f.write("# Elia Piccoli, Nicola Gugole\n")
        f.write("# ff15\n")
        f.write("# ML-CUP20 v1\n")
        f.write("# 15-06-2021\n")
        for i,o in enumerate(out):
            f.write(f"{i+1},{o[0]},{o[1]}\n")
    
if __name__=="__main__":
    ensemble_inference(None, plot=True)