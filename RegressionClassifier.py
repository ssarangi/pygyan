import numpy as np

class RegressionClassifier:
    def __init__(self, num_variables, learning_rate, threshold):
        self.num_variables = num_variables
        self.threshold = np.full(num_variables + 1, threshold)
        self.learning_rate = learning_rate

    # Training data is in the form of a numpy array with 2 dimensions.
    # y, x
    def set_training_data(self, training_data):
        self.num_training_data = len(training_data)
        self.training_data = training_data

        training_data_shape = self.training_data.shape
        if (len(training_data_shape) <= 1):
            raise Exception("Expected 2D training data set")

        if (training_data_shape[1] != self.num_variables + 1):
            raise Exception("Insufficient number of variables provided")

        # Add a column of 1's for theta0 just following y
        self.training_data = np.insert(self.training_data, 1, 1, axis=1)

    def set_initial_parameters(self, theta):
        self.theta = theta

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
