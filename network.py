import numpy as np

class Network():

    def __init__(self, layers, loss, loss_der):

        self.layers = layers
        self.loss = loss
        self.loss_der = loss_der

    # testing your model
    def predict(self, input_data):

        model_output = []
        for ii in range(len(input_data)):

            output = input_data[ii]
            for layer in self.layers:
                # the output of one layer becomes the input of the next
                output = layer.forward_propagation(output)
            # save final output 
            model_output.append(output)

        return model_output
    
    # training your model 
    # (updating weights and biases with backpropagation)
    def training(self, x_train, y_train, epochs, learning_rate):
        
        # epochs = number of times to show the model your data
        # AKA how many timesd it will update the weights and bias for each data point
        for ii in range(epochs):

            for xx in range(len(x_train)):
                
                # forward propagation with output of one layer
                # becomeing input of the next layer
                output = x_train[xx]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # perform backpropagation on this data point
                error = self.loss_der(y_train[xx], output)
                for layer in reversed(self.layers):
                    # update weigths and biases for each layer
                    error = layer.back_propagation(error, learning_rate)

            print(f'Epoch : {ii}')

    def training_mini(self, x_train, y_train, epochs, learning_rate, batch_size):

        # epochs = number of times to show the model your data
        # AKA how many timesd it will update the weights and bias for each data point
        for ii in range(epochs):

            for xx in range(0, len(x_train), batch_size):
                
                X_batch = x_train[xx:xx+batch_size]
                Y_batch = y_train[xx:xx+batch_size]

                # forward propagation with output of one layer
                # becoming input of the next layer
                err = []
                for xy in range(len(X_batch)):
                    
                    output = X_batch[xy]
                    for layer in self.layers:
                        output = layer.forward_propagation(output)

                    # perform backpropagation on this data point
                    error = self.loss_der(Y_batch[xy], output)

                    err.append(error)
                
                # Only backpropagate on the average error over the previous batch
                err = np.array(err)
                error = np.mean(err, axis=0)
                for layer in reversed(self.layers):
                    # update weigths and biases for each layer
                    error = layer.back_propagation(error, learning_rate)

            print(f'Epoch : {ii}')