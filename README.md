# NeuralNet-from-Scratch

This repositiory includes a Fully Connected (FC) Neural Network algorithm coded from scratch. The algorithm trains the model by taking in a set of input data and data labels, and then it can test the model by predicting the labels with a new set of data inputs. This algorithm uses gradient descent during backpropagation so the optimial weights and biases can be learned. 

Files:

`activationLayer.py` - activation layer class used to represent the functionalities of each activation layer, it takes in the desired activation function and the derivative of the activation function (for backpropagation)

    `forward_propagation` - performs forward propagation through the layer, takes in the input data

    `backward_propagation` - performs back propagation through the layer, takes in the error of the previous layer

`fullConnectedLayer.py` - fully connected layer class used to represent the functionalities of each FC layer, it takes in the desired input and output sizes for that layer. NOTE: the output size of one layer must be the input size of the next sequential layer.

    `forward_propagation` - performs forward propagation through the layer, takes in the input data

    `backward_propagation` - performs back propagation through the layer, takes in the error of the previous layer and model learning rate and updates the weights and biases of the layer

`functions.py` - contains various activation functions and thier derivatives for use in the activation layer. Also contains equations for an error metric and correlation coefficient.

`network.py` - network class used to train and test a neural network, it takes in an array of layers and a loss function and its derivative.

    `predict` - predicts the data label of given input data using the learned weights and biases from the trained model. It takes in the test input data

    `training` - trains the model via gradient descent (check out the write up to learn more). Takes in training data: x_train = model input data, y_train = data labels, number of epochs, and model learning rate

    `training_mini` - trains the model via a processes called mini batching where a set of data is forward propagated before the cumulative error is backpropagated. Takes in training data: x_train = model input data, y_train = data labels, number of epochs, model learning rate, and batch size

Checkout `NNWriteUp.pdf` for detailed explanation of a Neural Network and its functionalities