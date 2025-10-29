import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=10e-5, epochs=1, alpha=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None #This will be initialized as an array of zeros inside the fit method based on the input data dimensions
        self.bias = 0
        self.alpha = alpha

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
                weights_gradient = (features * error) + (self.weights * self.alpha)
                self.bias = self.bias - (bias_gradient * self.learning_rate)
                self.weights = self.weights - (weights_gradient * self.learning_rate)
        return self

    def fit_normal(self, training_data, target_values):
        bias_column = np.ones(training_data.shape[0])
        updated_train = np.concatenate((np.expand_dims(bias_column, axis=1), training_data), axis=1)
        identity = np.identity(updated_train.shape[1])
        identity[:,0] = 0
        l2_term = (self.alpha * identity) 
        theta = np.linalg.inv(updated_train.T @ updated_train + l2_term) @ updated_train.T @ target_values
        self.bias = theta[0]
        self.weights = theta[1:]
        return self
