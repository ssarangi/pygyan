from LinearRegressionClassifier import LinearRegressionClassifier
import numpy as np

def generate_linear_regression_one_variable_test_data(theta0, theta1):
    # We will generate an equation of a line and then generate points based on it
    x = np.arange(-100, 100.5, 0.5) # Generate a bunch of points from -100 to 100 with increments of 0.5
    y = theta0 + theta1 * x

    training_data = np.vstack((y, x)).T
    print(training_data)
    return training_data


def main():
    theta0 = 4
    theta1 = 1
    training_data = generate_linear_regression_one_variable_test_data(theta0 = theta0, theta1 = theta1)

    initial_theta0 = 5
    initial_theta1 = 4

    num_variables = 1
    learning_rate = 0.0001
    threshold     = 0.0000001

    linear_regression_classifier = LinearRegressionClassifier(num_variables = 1, learning_rate = learning_rate, threshold = threshold)
    linear_regression_classifier.set_initial_parameters(theta0 = initial_theta0, theta1 = initial_theta1)
    linear_regression_classifier.set_training_data(training_data)
    linear_regression_classifier.gradient_descent(original_theta0=theta0, original_theta1=theta1)
    linear_regression_classifier.plot_contour_graph()

if __name__ == "__main__":
    main()
