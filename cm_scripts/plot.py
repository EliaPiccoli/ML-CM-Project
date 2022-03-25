import numpy as np
import matplotlib.pyplot as plt

def plot_single_model(cup_model, fstar, axs, color, label):
    """ function to generate the convergence rate, log residual rate and residual rate of a given model 
        optimization process, given an fstar

    Args:
        cup_model (svr model object): containing history for all function values during fitting
        fstar (float): value necessary for conv rate, log res error and res error computation
        axs (plt axis object): needed for plotting
        color (string): to define which color to assign to plotted graphs
        label (string): to define name of plotted curves

    Returns:
        plot_conv_rate: list of conv rate computed over all function values during fitting
        log_residual_error: list of log residual error computed over all function values during fitting
        residual_error: list of residual error computed over all function values during fitting
    """    
    # set up variables for plotting
    conv_rate_threshold_noise = 100
    plot_conv_rate = []
    residual_error = []
    log_residual_error = []
    for i in range(len(cup_model.history['f']) - 1):
        if i < len(cup_model.history['f']) - conv_rate_threshold_noise:
            plot_conv_rate.append((cup_model.history['f'][i+1] - fstar) / (cup_model.history['f'][i] - fstar))
        log_residual_error.append(np.log(np.abs(cup_model.history['f'][i] - fstar) / np.abs(fstar)))
        residual_error.append(np.abs(fstar - cup_model.history['f'][i]) / np.abs(fstar))
    # plot results
    axs[0].plot(range(len(plot_conv_rate)), plot_conv_rate, label=label, color=color)
    axs[0].set_ylabel("CONV_RATE")
    axs[1].plot(range(len(log_residual_error)), log_residual_error, label=label, color=color)
    axs[1].set_ylabel("LOG_RESIDUAL_ERROR")
    axs[2].plot(range(len(residual_error)), residual_error, label=label, color=color)
    axs[2].set_ylabel("RESIDUAL_ERROR")
    axs[0].set_ylim(0.5, 1.5)  # to avoid artifacts in plot
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    return plot_conv_rate, log_residual_error, residual_error

def plot_svr_predict(x, y, pred, text="fig_title"):
    """function to plot model predictions (in blue) over ground truths (in red), for all input dimensions

    Args:
        x (tensor): input of task, important also for getting the input dimensionality
        y (tensor): output of the task, relevant for the ground truth
        pred (tensor): output predicted by the model
        text (str, optional): title of the plot. Defaults to "fig_title".
    """    
    fig,axs = plt.subplots(2,5, figsize=(15,15))
    for i in range(x.shape[1]):
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],y,color="red",marker='x')
        axs[i//(x.shape[1]//2)][i%(x.shape[1]//2)].scatter(x[:,i],pred,color="blue",marker='.')
    fig.suptitle(text)
    plt.show()