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

        label = "Accuracy" if classification else "MEE"
            
        fig = plt.figure()
        fig.canvas.set_window_title(title)
        outer = gridspec.GridSpec(subplot_size[0], subplot_size[1], wspace=0.2, hspace=0.2)
        fig.tight_layout()
        for i in range(subplot_size[0]*subplot_size[1]):

            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)

            
            for j in range(2):
                ax = plt.Subplot(fig, inner[j])
                ax.set_xlabel("Epoch" if j == 1 else "")
                ax.set_ylabel(label if j == 0 else "Loss")
                if j == 0:
                    ax.set_xticks([])
                ax.plot(range(epochs[i]), [stat[2*j] for stat in stats[i]], label="Train")
                ax.plot(range(epochs[i]), [stat[2*j+1] for stat in stats[i]], label="Validation", linestyle='dashed')
                ax.legend()
                fig.add_subplot(ax)
            
        plt.show(block=block)
