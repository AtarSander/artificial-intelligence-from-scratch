import matplotlib.pyplot as plt
import numpy as np


def plot_results(experiments_table, g_mean, g_stats, g_stats_std):
    ind = experiments_table["Individuals number"]
    batch_size = len(ind)/len(set(ind))
    max_indexes = [np.argmax(g_mean[i:i+int(batch_size)])+i
                   for i in range(0, len(g_mean), int(batch_size))]
    for i in max_indexes:
        timesteps = np.linspace(0, 1000,
                                experiments_table["Iterations number"][i])
        legend = f"""\u03BC={experiments_table["Individuals number"][i]} \
                    T_max={experiments_table["Iterations number"][i]} \
                    pc={experiments_table["Cross probability"][i]} \
                    pm={experiments_table["Mutation probability"][i]}"""
        plt.plot(timesteps, g_stats[i], label=legend)
        plt.fill_between(timesteps, g_stats[i] - g_stats_std[i],
                         g_stats[i] + g_stats_std[i], alpha=0.2)

    plt.xlabel("Iteration")
    plt.ylabel("Target function value")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_example(g_run, t_steps, legend):
    timesteps = np.linspace(0, t_steps, t_steps)
    plt.plot(timesteps, g_run, label=legend)
    plt.xlabel("Iteration")
    plt.ylabel("Target function value")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()
