import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class Training(object):
    """
    Training class for NeuralNetwork

    this class trains and updates the layers in the NeuralNetwork.

    ...

    Attributes
    ----------
    NeuralNetwork: list
        a list of NeuralLayer classes in sequential order
    Data: Data class
        a dataclass which implements a get_next_minibatch method
    learning_rate: float
        learning rate
    epoch: int
        number of epochs
    l2_reg_lambda: float
        lambda for controlling l2 regularization

    """

    def __init__(self, NeuralNetwork, Data, learning_rate, epochs, l2_reg_lambda):
        self.NeuralNetwork = NeuralNetwork
        self.Data = Data
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2_reg_lambda = l2_reg_lambda
        self.cross_entropy_loss = None
        self.train_loss_history = []
        self.test_loss_history = []

    def train(self):
        """
        Training of the NeuralNetwork,

        loop with forwardprop, loss, backwardprop and parameter update

        :return: None
        """
        for epoch in range(0, self.epochs):

            for x_train, y_train in self.Data.get_next_minibatch():
                self.forward_propagation(x_train)
                delta_loss = self.compute_cross_entropy_loss(y_train=y_train, derivative=True)
                self.backward_propagation(delta_loss)
                self.update_parameters()

            if epoch % (self.epochs * 0.01) == 0:
                self.train_loss_history.append(self.test_prediction(self.Data.X_val, self.Data.y_val))
                self.test_loss_history.append(self.test_prediction(self.Data.X_test, self.Data.y_test))

    def forward_propagation(self, x_training):
        """
        Forward propagation of NeuralNetwork (assumes sequential order of layers)
        :param x_training: training data sample
        :return: activation of last layer
        """
        self.A_curr = x_training
        for layer in self.NeuralNetwork:
            A_prev = self.A_curr
            self.A_curr = layer.layer_forward_propagation(A_prev)
        return self.A_curr

    def backward_propagation(self, delta_loss):
        """
        Backward propagation of NeuralNetwork (assumes sequential order of layers)
        :param delta_loss: lossfunction derivative
        :return: None
        """
        dA_prev = delta_loss
        for layer in reversed(self.NeuralNetwork):
            dA_curr = dA_prev
            dA_prev = layer.layer_backward(dA_curr)

    def update_parameters(self):
        '''
        update the parameters in the layers
        :return: None
        '''
        for layer in self.NeuralNetwork:
            layer.update_parameters(learning_rate=self.learning_rate,
                                    l2_reg_lambda=self.l2_reg_lambda)

    def test_prediction(self, x_test, y_test):
        """
        Test Prediction of the Network
        :param x_test: x_test dataset
        :param y_test: y_test dataset
        :return: cross entropy loss
        """
        self.pred = self.forward_propagation(x_test)
        return self.cross_entropy(Y_hat=self.pred, Y=y_test)

    def test_report(self):
        """
        Generate TestReport from the Network
        :return:
        """
        pred = self.forward_propagation(self.Data.X_test)
        converted_pred = (pred >= 0.5).astype(np.int) # 0.5 considered 1
        return classification_report(y_true=self.Data.y_test, y_pred=converted_pred)

    def mathplot_training_test_loss(self):
        """
        Generate plot of train and testloss
        :return: show plot
        """
        epoch_count = range(1, len(self.train_loss_history) + 1)
        plt.plot(epoch_count, self.train_loss_history, 'r--')
        plt.plot(epoch_count, self.test_loss_history, 'b-')
        plt.legend(['Training loss', 'Test Loss'])
        plt.xlabel('Epoch')
        plt.xlabel('Loss')
        plt.show()

    def cross_entropy(self, Y_hat, Y):
        """
        computes cross entropy loss

        :param Y_hat: prediction distribution
        :param Y: True distribution
        :return: cross entropy loss
        """
        m = Y_hat.shape[0]
        E = 0
        result = []
        for i in range(0, m):
            if Y[i] == 1:
                E -= np.log(Y_hat[i])
            else:
                E -= np.log(1 - Y_hat[i]) # RuntimeWarning divide by zero encountered
            result.append(E)
            E = 0
        return np.array(1 / m * sum(result))

    def compute_cross_entropy_loss(self, y_train, derivative=False):
        """
        Computes cross entropy loss or deriviative os binary cross entropy derivative

        :param y_train: y training dataset
        :param derivative:  derivative of binary cross entropy
        :return: cross entropy / binary cross entropy derivative
        """
        if derivative:
            return (self.A_curr.T - y_train).T  # assumes binary cross entropy
        return self.cross_entropy(self.A_curr, y_train)