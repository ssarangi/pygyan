import numpy as np

class LogisticRegressionClassifier:
    def __init__(self, learning_rate, threshold):
        self.num_vars = 1
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.theta0 = 0
        self.theta1 = 0
        self.cost_func = []
        self.theta0_iter = []
        self.theta1_iter = []


