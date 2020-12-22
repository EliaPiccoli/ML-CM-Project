import matplotlib.pyplot as plt

class Plot:
    # utilities method for plotting train/test results
    @staticmethod
    def _plot_train_stats(stats, title = 'Training Statistics', epochs = 200, subp_len = 2):
        # stats semantics as follows: 0 - TrainAcc | 1 - ValAcc | 2 - TrainLoss | 3 - ValLoss | more can be added..
        fig, axs = plt.subplots(subp_len)
        fig.canvas.set_window_title(title)

        for i in range(subp_len):
            axs[i].set_xlabel("Epoch")
            axs[i].set_ylabel("Accuracy" if i == 0 else "Loss")
            axs[i].plot(range(epochs), [stat[2*i] for stat in stats], label="Train")
            axs[i].plot(range(epochs), [stat[2*i+1] for stat in stats], label="Validation")
            axs[i].legend()

        plt.tight_layout()
        plt.show()
