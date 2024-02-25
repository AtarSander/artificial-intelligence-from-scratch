import matplotlib.pyplot as plt
import numpy as np


def plot_results(experiments_table, g_mean, g_stats, g_stats_std):
    ind = experiments_table["Individuals number"]
    batch_size = len(ind) / len(set(ind))
    max_indexes = [
        np.argmax(g_mean[i : i + int(batch_size)]) + i
        for i in range(0, len(g_mean), int(batch_size))
    ]
    for i in max_indexes:
        timesteps = np.linspace(0, 1000, experiments_table["Iterations number"][i])
        legend = f"""\u03BC={experiments_table["Individuals number"][i]} \
                    T_max={experiments_table["Iterations number"][i]} \
                    pc={experiments_table["Cross probability"][i]} \
                    pm={experiments_table["Mutation probability"][i]}"""
        plt.plot(timesteps, g_stats[i], label=legend)
        plt.fill_between(
            timesteps,
            g_stats[i] - g_stats_std[i],
            g_stats[i] + g_stats_std[i],
            alpha=0.2,
        )

    plt.title("Results of each generation for the best set of parameters")
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


def plot_bar(experiments_table, g_mean, parameters, shortcuts, budget=4000):
    values = []
    values_std = []
    categories = []

    for parameter in parameters:
        for value in set(experiments_table[parameter]):
            results = []
            for j in enumerate(experiments_table[parameter]):
                if j[1] == value:
                    results.append(g_mean[j[0]])
            values.append(np.mean(results))
            values_std.append(np.std(results))
            if parameter == "Individuals number":
                categories.append(
                    shortcuts[parameter]
                    + "="
                    + str(value)
                    + ", "
                    + shortcuts["t_max"]
                    + "="
                    + str(budget // value)
                )
            else:
                categories.append(shortcuts[parameter] + "=" + str(value))

    plt.bar(categories, values, yerr=values_std, align="center", ecolor="black")
    plt.xticks(rotation="vertical")
    plt.xlabel("Parameters")
    plt.ylabel("Mean results")
    plt.title("How each parameter did for given value")
    plt.tight_layout()
    plt.show()
