import numpy as np

class FullConnected():
    """ Class for the Full Connected layer of the network

    performs forward and packward propagation of the layer
    where weights and biases are updated

    """

    def __init__(self, input_size, output_size):

        # initialize weights and biases from -0.5 to 0.5
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # computes output of layer y = w*x + b given input data
    def forward_propagation(self, input_data):
        self.input_data = input_data
        self.output = np.dot(self.input_data, self.weights) + self.bias

        return self.output
    
    # finds the gradients of error and updates weights and biases accordingly
    def back_propagation(self, error, learning_rate):
        error_x = np.dot(error, self.weights.T)
        error_w = np.dot(self.input_data.T, error)
        error_b = error

        # update w & b
        self.weights -= learning_rate*error_w
        self.bias -= learning_rate*error_b

        # return error_x for use in other layer backpropagation
        return error_x

