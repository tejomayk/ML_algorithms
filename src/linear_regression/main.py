import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=10e-5, epochs=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None #This will be initialized as an array of zeros inside the fit method based on the input data dimensions
        self.bias = 0

    def predict(self, input_data: np.array):
        output_value = np.dot(input_data, self.weights) + self.bias