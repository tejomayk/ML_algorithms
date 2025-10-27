import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=10e-5, epochs=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None #This will be initialized as an array of zeros inside the fit method based on the input data dimensions
        self.bias = 0

    def predict(self, input_data: np.array):
        output_value = np.dot(input_data, self.weights) + self.bias
        return output_value
    
    def fit(self, training_data: np.array, target_values):
        self.weights = np.zeros(training_data.shape[1])
        for _ in range(self.epochs):
            for features, value in zip(training_data, target_values):
                prediction = self.predict(features)
                error = prediction - value
                bias_gradient = error
                weights_gradient = features * error
                self.bias = self.bias - (bias_gradient * self.learning_rate)
                self.weights = self.weights - (weights_gradient * self.learning_rate)
