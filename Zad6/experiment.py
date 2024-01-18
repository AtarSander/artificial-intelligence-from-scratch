from model import Qmodel
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt
import numpy as np


def experiment(env, learning_rates, gammas, epsilons, t_maxes, number_episodes):
    qmodel = Qmodel(env)
    results = {}
    results["Number of episodes"] = number_episodes
    results["Learning rate"] = learning_rates
    results["Discount factor"] = gammas
    results["Exploration probability"] = epsilons
    results["Max episode steps"] = t_maxes
    rewards = []
    steps = []
    max_reward = 0
    best_model_dict = {}
    best_q_table, accuracies = [], []
    for i in range(len(learning_rates)):
        accuracy = qmodel.train(
            learning_rates[i],
            gammas[i],
            epsilons[i],
            t_maxes[i],
            number_episodes[i],
        )
        rewards.append(accuracy["Average_reward"][(number_episodes[i] // 100)-2])
        steps.append(
            accuracy["Average_steps_per_episode"][(number_episodes[i] // 100)-2]
        )
        accuracies.append(accuracy)
        if accuracy["Average_reward"][(number_episodes[i] // 100)-2] > max_reward:
            max_reward = accuracy["Average_reward"][(number_episodes[i] // 100)-2]
            best_model_dict = accuracy
            best_q_table = qmodel.get_params()["Q_table"]
    results["Final reward"] = rewards
    results["Final steps number"] = steps
    return results, best_model_dict, best_q_table, accuracies


def visualize(env, q_table, episodes):
    for i in range(episodes):
        end = False
        state, _ = env.reset()
        while not end:
            action = np.argmax(q_table[state, :])
            state, reward, end, _, _ = env.step(action)
            screen = env.render()
            for text in plt.gca().texts:
                text.remove()
            plt.imshow(screen)
            plt.title(f"Episode: {i+1}")
            plt.text(
                0.5,
                -0.1,
                f"State: {state}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )
            plt.text(
                0.5,
                -0.2,
                f"Action: {action}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )
            plt.text(
                0.5,
                -0.3,
                f"Reward: {reward}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )
            plt.xticks([])
            plt.yticks([])
            ipythondisplay.clear_output(wait=True)
            ipythondisplay.display(plt.gcf())

    ipythondisplay.clear_output(wait=True)
    env.close()


def plot_results(accuracies):
    for i in range(len(accuracies)):
        plt.plot(
            accuracies[i]["Episode_number"],
            accuracies[i]["Average_reward"],
            label=f"Model: {i}"
        )
    plt.title(f"Average Reward over Episodes in models")
    plt.xlabel("Episode Number")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.show()
