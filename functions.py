import numpy as np
from scipy import stats

# activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

# need to determine the proper derivative of relu
def relu(x):
    return max(0.0, x)

# def relu_derivative(x):
#     return 

# loss function and their derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred,2))

def mse_derivative(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

# calculating important statistics for evaluating the model
def r_sq_stats(y_true, y_pred):
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    r_sq = r_value**2
    return r_sq, std_err