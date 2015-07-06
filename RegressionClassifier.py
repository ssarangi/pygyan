import numpy as np

class RegressionClassifier:
    def __init__(self, num_variables, learning_rate, threshold):
        self.num_variables = num_variables
        self.threshold = threshold
        self.learning_rate = learning_rate