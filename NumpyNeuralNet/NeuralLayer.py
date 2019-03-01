import numpy as np


class NeuralLayer(object):
    """
    Neural Network layer class

    Implements a Neural Network layer with inputs, units and activationfunction.
    Computes FeedForward, Backpropagation and parameterupdates for the layer, these layers
    are then combined in sequential order in a list to create a neural "network"

    ...

    Attributes
    ----------
    inputs: int
        number of inputs to the neuron layer
    units: int
        number of units in the neuron layer
    activation: str
        string with the name of the activationfunction ('relu' or 'sigmoid')

    """

    def __init__(self, inputs, units, activation):
        self.inputs = inputs
        self.units = units
        self.activation = activation
        self.d_weights = None
        self.weights = None
        self.biases = None
        self.A_prev = None
        self.Z_curr = None
        self.A_curr = None
        self.d_biases = None
        self.activation_function = None

    def init_layer(self):
        """ initialize layer """

        np.random.seed(seed=99)
        self.init_weight_matrix()
        self.init_bias_matrix()
        self.init_activation_function()

    def init_weight_matrix(self):
        """ initialize weights"""
        self.weights = np.random.randn(self.inputs, self.units) * 0.1

    def init_bias_matrix(self):
        """ initialize bias"""
        self.biases = np.random.randn(1, self.units) * 0.1

    def init_activation_function(self):
        """ initialize activationfunction"""
        if self.activation == 'relu':
            self.activation_function = self.relu
        elif self.activation == 'sigmoid':
            self.activation_function = self.sigmoid

    def layer_forward_propagation(self, A_prev):
        """
        computes forward propagation with previous layer activationfunction output
        :param A_prev: Previous layer activations or training data if current layer is the first
        :return: Current layer activation
        """
        self.A_prev = A_prev
        self.Z_curr = np.dot(self.A_prev, self.weights) + self.biases
        self.A_curr = self.activation_function(self.Z_curr)
        return self.A_curr

    def layer_backward(self, dA_prev):
        """
        computes backward propagation with next layers derivative
        :param dA_prev: Next layers derivative, loss error if current layer is the last
        :return: Current layer derivative
        """
        self.d_weights = np.dot(self.A_prev.T, dA_prev)
        self.d_biases = np.sum(dA_prev, axis=0)
        dA_curr = np.dot(dA_prev, self.weights.T) * self.activation_function(self.A_prev, derivative=True)
        return dA_curr

    def l2_reg(self, l2_reg_lambda):
        """
        Adds L2 Regularization to the layers derivative of weights matrix,
        :param l2_reg_lambda: lambda for L2 regularization
        :return: l2 regularization updated derivative matrix
        """
        self.d_weights += l2_reg_lambda * self.weights
        return self.d_weights

    def update_parameters(self, learning_rate, l2_reg_lambda):
        """
        Update layer parameters (weights and biases)
        :param learning_rate: learningrate for the layer
        :param l2_reg_lambda: l2 regularization lambda for the layer
        :return: None
        """
        self.weights += -learning_rate * self.l2_reg(l2_reg_lambda=l2_reg_lambda)
        self.biases += -learning_rate * self.d_biases

    def sigmoid(self, X, derivative=False):
        """
        Computes sigmoid on X with fixed interval to prevent under or overflow
        :param X: neuronlayer output
        :param derivative: bool if derivative
        :return: sigmoid of X
        """
        # Prevent overflow.
        X = np.clip(X, -500, 500)

        # Calculate activation signal
        X = 1.0 / (1 + np.exp(-X))

        if derivative:
            # Return the partial derivation of the activation function
            return np.multiply(X, 1 - X)
        else:
            # Return the activation signal
            return X

    def relu(self, X, derivative=False):
        """
        Computes relu on X
        :param X:  neuron layer output
        :param derivative: bool if derivative
        :return: relu of X
        """
        if derivative:
            X[X <= 0] = 0
            X[X > 0] = 1
            return X
        return np.maximum(0, X)