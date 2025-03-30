# Artificial inteligence algorithms from scratch
This repository contains from-scratch implementations of seven algorithms from various branches of artificial intelligence. Implementations are tested and evaluated on different tasks with in-depth analysis of used hyperparameters and general conclussions about the algorithm in the form of report. Each algorithm is a case study of different branch of AI, including:
| Branch | Algorithm |
| --------- | --------- |
| Optimization | gradient descent |
| Evolution algorithms | genetic algorithm |
| Two-player zero-sum games | alpha beta pruning |
| Machine learning, classification | ID3 tree |
| Deep learning | feedforward neural net |
| Reinforcement learning | Q-learning |
| Probablistic classifier | naive bayes classifier |
# Tech stack
In order to understand and evaluate the inner workings of the algorithms, all of them are implemented almost exclusively with numpy. Additional libraries are used for data preprocessing, setting up testing environment and/or results visualization.
### Core libraries:
- Numpy
- Matplotlib
- Pandas
### Auxiliary libraries:
| Algorithm | Libraries |
| --------- | --------- |
| Gradient descent | mpl_toolkits |
| ID3 classifier | sklearn, seaborn |
| Neural network | keras.datasets, PIL |
| Q-learning | IPython, gymnasium |
| Naive bayes | sklearn, seaborn |
# Data
In order to test and evaluate accuracy of certain algorithms it was necessary to use external datasets.
| Algorithm | Dataset |
| --------- | --------- |
| ID3 classifier | https://www.kaggle.com/datasets/bhadaneeraj/cardio-vascular-disease-detection |
| Neural network | http://yann.lecun.com/exdb/mnist/ |
| Q-learning | https://gymnasium.farama.org/environments/toy_text/taxi/ |
| Naive bayes | https://www.kaggle.com/datasets/bhadaneeraj/cardio-vascular-disease-detection |
