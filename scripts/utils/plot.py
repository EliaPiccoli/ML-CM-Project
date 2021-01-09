import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Plot:
    # utilities method for plotting train/test results
    @staticmethod
    def _plot_train_stats(stats, title = 'Training Statistics', epochs = 200, max_graphs_per_row = 4, block=True, classification = True):
        # stats semantics as follows: 0 - TrainAcc | 1 - ValAcc | 2 - TrainLoss | 3 - ValLoss | more can be added..
        if len(stats) <= max_graphs_per_row:
            subplot_size = (1,len(stats))
        else:
            subplot_size = (int(len(stats) / max_graphs_per_row), max_graphs_per_row)
            
        fig = plt.figure()
        fig.canvas.set_window_title(title)
        outer = gridspec.GridSpec(subplot_size[0], subplot_size[1], wspace=0.2, hspace=0.2)
        fig.tight_layout()
        for i in range(subplot_size[0]*subplot_size[1]):
            x = 2 if classification else 1

            inner = gridspec.GridSpecFromSubplotSpec(x, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)

            
            for j in range(x):
                if classification:
                    ax = plt.Subplot(fig, inner[j])
                    ax.set_xlabel("Epoch" if j == 1 else "")
                    ax.set_ylabel("Accuracy" if j == 0 else "Loss")
                    if j == 0:
                        ax.set_xticks([])
                    ax.plot(range(epochs[i]), [stat[2*j] for stat in stats[i]], label="Train")
                    ax.plot(range(epochs[i]), [stat[2*j+1] for stat in stats[i]], label="Validation")
                    ax.legend()
                    fig.add_subplot(ax)
                else:
                    ax = plt.Subplot(fig, inner[j])
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.plot(range(epochs[i]), [stat[2] for stat in stats[i]], label="Train") # train loss
                    ax.plot(range(epochs[i]), [stat[3] for stat in stats[i]], label="Validation") # val loss
                    ax.legend()
                    fig.add_subplot(ax)
            
        plt.show(block=block)
