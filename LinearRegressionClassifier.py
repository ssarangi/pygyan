# Project Imports
from RegressionClassifier import RegressionClassifier

import numpy as np
import math
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

class LinearRegressionClassifier(RegressionClassifier):
    def __init__(self, num_variables, learning_rate, threshold):
        RegressionClassifier.__init__(self, num_variables=num_variables, learning_rate=learning_rate, threshold=threshold)
        self.theta0 = 0
        self.theta1 = 0
        self.cost_func = []
        self.theta0_iter = []
        self.theta1_iter = []
        self.theta = np.zeros(num_variables)

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

    def compute_func(self, theta, x):
        result = 0
        if (theta.shape[0] != x.shape[0]):
            raise Exception("Unexpected shape of theta or x parameters")

        per_element_sum = theta * x
        result = np.sum(per_element_sum)
        return result

    def compute_cost_func(self, theta):
        cost = 0
        for i, ith_training_data in enumerate(self.training_data):
            y_i = ith_training_data[0]
            x_i = ith_training_data[1:]
            cost += pow((self.compute_func(theta, x = x_i) - y_i), 2)

        cost = (1 / (2 * self.num_training_data)) * cost
        return cost

    def plot_contour_graph(self):
        cost_values = np.zeros((100, 100))

        min_theta0 = min(self.theta0_iter)
        max_theta0 = max(self.theta0_iter)

        min_theta1 = min(self.theta1_iter)
        max_theta1 = max(self.theta1_iter)

        theta0_mg = np.linspace(min_theta0, max_theta0, 100)
        theta1_mg = np.linspace(min_theta1, max_theta1, 100)

        for i, t0 in enumerate(theta0_mg):
            for j, t1 in enumerate(theta1_mg):
                cost = self.compute_cost_func(theta0 = t0, theta1 = t1)
                cost_values[i, j] = cost

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        surf = ax.plot_surface(theta0_mg, theta1_mg, cost_values, cmap=cm.coolwarm, antialiased=False)

        fig.colorbar(surf)
        plt.show()

    def gradient_descent(self, original_theta0, original_theta1):
        # Set the initial difference to max infinity
        theta0_diff = sys.maxsize
        theta1_diff = sys.maxsize
        theta_diff = np.full(len(self.theta), sys.maxsize)

        iteration = 0

        print("Initial Theta0: %s         Initial Theta1: %s" % (self.theta0, self.theta1))

        # Reset the theta0, theta1 and cost function calculations
        self.theta0_iter = []
        self.theta1_iter = []
        self.cost_func   = []

        while (np.greater(theta_diff, self.threshold).any()):
            derivative_theta_i = np.zeros(len(self.theta))

            for i, ith_training_data in enumerate(self.training_data):
                y_i = ith_training_data[0]
                x_i = ith_training_data[1:]
                derivative_theta_i += x_i * (self.compute_func(theta = self.theta, x = x_i) - y_i)

            theta_new = self.theta - (1 / self.num_training_data) * self.learning_rate * derivative_theta_i

            theta_diff = abs(theta_new - self.theta)

            # self.theta0_iter.append(self.theta0)
            # self.theta1_iter.append(self.theta1)
            # Update the cost function
            self.cost_func.append(self.compute_cost_func(self.theta))

            self.theta = theta_new

            print("Iteration: %s Theta0: %s Theta1: %s" % (iteration, self.theta[0], self.theta[1]))
            iteration += 1

        print("Original: theta0: %s          theta1: %s" % (original_theta0, original_theta1))
        print("My Guess: theta0: %s          theta1: %s" % (self.theta[0], self.theta[1]))