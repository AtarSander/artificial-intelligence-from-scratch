import gymnasium as gym


class Qmodel:
    def __init__(self):
        self.env = gym.make("Taxi-v3")

    def get_params(self):
        pass

    def train(self, learning_rate, num_episodes):
        pass
