import numpy as np
import random


class Qmodel:
    def __init__(self, env):
        self.env = env
        self.trained = False

    def get_params(self):
        params = {}
        if self.trained:
            params["Learning_rate"] = self.beta
            params["Discount_factor"] = self.gamma
            params["Exploration_probability"] = self.epsilon
            params["Number_of_episodes"] = self.num_episodes
            params["Max_episode_length"] = self.t_max
            params["Q_table"] = self.q_table
        return params

    def set_q_table(self, q_table):
        self.q_table = q_table

    def train(self, beta, gamma, epsilon, t_max, num_episodes):
        self.trained = True
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.t_max = t_max
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        accuracy = {}
        numbers = []
        rewards = []
        steps = []
        penalties = []
        for i in range(num_episodes):
            current_state, _ = self.env.reset()
            end = False
            t = 0

            while not end and t < t_max:
                if random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[current_state, :])

                next_state, reward, end, _, _ = self.env.step(action)
                delta = (
                    reward
                    + gamma * np.max(self.q_table[next_state, :])
                    - self.q_table[current_state, action]
                )
                self.q_table[current_state, action] += beta * delta
                t += 1
                current_state = next_state
            if i % 1000 == 0:
                total_rewards, total_steps, total_penalties = self.evaluate(100)
                numbers.append(i)
                rewards.append(total_rewards)
                steps.append(total_steps)
                penalties.append(total_penalties)

        accuracy["Episode_number"] = numbers
        accuracy["Average_reward"] = rewards
        accuracy["Average_steps_per_episode"] = steps
        accuracy["Penalties"] = penalties
        return accuracy

    def evaluate(self, eval_episodes_num):
        total_reward = 0
        total_steps = 0
        total_penalties = 0
        for _ in range(eval_episodes_num):
            current_state, _ = self.env.reset()
            end = False
            t = 0
            penalties = 0
            while not end and t < self.t_max:
                action = np.argmax(self.q_table[current_state, :])
                current_state, reward, end, _, _ = self.env.step(action)
                total_reward += reward
                t += 1
                if reward == -10:
                    penalties += 1
            total_steps += t
            total_reward /= t
            total_penalties += penalties
        total_reward /= eval_episodes_num
        total_steps /= eval_episodes_num
        total_penalties /= eval_episodes_num
        return total_reward, total_steps, total_penalties
