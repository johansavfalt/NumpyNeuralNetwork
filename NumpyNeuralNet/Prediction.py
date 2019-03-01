from NeuralLayer import NeuralLayer
from Data import make_moons_helper
from Training import Training


def main():
    """
    Creates NeuralNetwork and Data class

    train NeuralNetwork on data and print testreport

    :return: None
    """

    # Create NeuralNetwork by combining NeuralLayer class in list
    print('creating network ....')
    NeuralNetwork = [
        NeuralLayer(2, 4, activation='relu'),
        NeuralLayer(4, 6, activation='relu'),
        NeuralLayer(6, 4, activation='relu'),
        NeuralLayer(4, 1, activation='sigmoid')
    ]

    # Initialize layers
    print('initializing network layers ....')
    for layer in NeuralNetwork:
        layer.init_layer()

    # make_moons data set from sklearn, with custom helper class
    print('create example dataset "make_moons" from sklearn ....')
    make_moons_data = make_moons_helper(n_samples=1000,
                                        noise=0.2,
                                        test_size=0.1,
                                        eval_size=0.1,
                                        minibatch_size=25)
    print('create training instance ....')
    # Create Training instance with NeuralNetwork, Data and Parameters
    training = Training(NeuralNetwork=NeuralNetwork,
                        Data=make_moons_data,
                        epochs=1000,
                        learning_rate=0.01,
                        l2_reg_lambda=0.001)
    # Train
    print('training network ....')
    training.train()

    # Classification Report
    print('result: ')
    print(training.test_report())

    # Training and Test loss plot
    # training.mathplot_training_test_loss()


if __name__ == '__main__':
    main()