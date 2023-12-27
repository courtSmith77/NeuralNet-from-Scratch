import numpy as np

class Activation():
    """ Class for the Activation layer of the network

    performs forward and packward propagation of the layer
    where the inputs and outputs are changed to be within a
    range specified by the activation function (usually between
    (-1,1) or (0,1))

    *** note:
    made forward and backward propagation the same form in both
    activation and full connected layer class so it's uniform to call

    """

    def __init__(self, act_func, act_derivative):

        # saves the function to be used for activation
        self.activation_func = act_func
        self.activation_derivative = act_derivative

    # forward prop returns the activated point
    def forward_propagation(self, input_data):
        
        self.input_data = input_data
        self.output = self.activation_func(self.input_data)

        return self.output
    
    # back prop uses derivative of activation to find 
    # gradient of error of activation layer 
    # does not use learning_rate but has for consistency with fullconnected layer
    def back_propagation(self, error, learning_rate):

        output = self.activation_derivative(self.input_data)*error

        return output



