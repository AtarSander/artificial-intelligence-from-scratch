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
    best_q_table = []
    for i in range(len(learning_rates)):
        accuracy = qmodel.train(
            learning_rates[i],
            gammas[i],
            epsilons[i],
            t_maxes[i],
            number_episodes[i],
        )
        rewards.append(accuracy["Average_reward"][(number_episodes[i] // 1000) - 1])
        steps.append(
            accuracy["Average_steps_per_episode"][(number_episodes[i] // 1000) - 1]
        )
        if accuracy["Average_reward"][(number_episodes[i] // 1000) - 1] > max_reward:
            max_reward = accuracy["Average_reward"][(number_episodes[i] // 1000) - 1]
            best_model_dict = accuracy
            best_q_table = qmodel.get_params()["Q_table"]
    results["Final reward"] = rewards
    results["Final steps number"] = steps
    return results, best_model_dict, best_q_table


def visualize(env, q_table, episodes):
    state, _ = env.reset()
    for i in range(episodes):
        end = False
        while not end:
            action = np.argmax(q_table[state, :])
            state, _, end, _, _ = env.step(action)
            screen = env.render()
            plt.imshow(screen)
            plt.title(f"Episode{i+1}")
            ipythondisplay.clear_output(wait=True)
            ipythondisplay.display(plt.gcf())

    ipythondisplay.clear_output(wait=True)
    env.close()
