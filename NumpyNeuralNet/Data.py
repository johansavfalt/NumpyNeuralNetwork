import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


class make_moons_helper(object):
    """
    sklearn dataset "make_moons" helper class

    ...
    Attributes
    ----------

    n_samples: int
        total number of generated samples

    noise: float
        noise in the data

    random_state: int
        random_state

    test_size: float
        test set size

    eval_size: float
        evalutation set size

    minibatch_size: int
        minibatch size

    """

    def __init__(self, n_samples=1000, noise=0.2, test_size=0.1, eval_size=0.1, minibatch_size=25):

        # make moons dataset
        self.X, self.y = make_moons(n_samples=n_samples, noise=noise, random_state=100)

        # create testset
        self.X_train, self.X_test, self.y_train, self.y_test \
            = train_test_split(self.X, self.y, test_size=test_size, random_state=42)

        # create train and validation set

        self.X_train, self.X_val, self.y_train, self.y_val \
            = train_test_split(self.X_train, self.y_train, test_size=eval_size, random_state=42)

        # minibatch_size
        self.minibatch_size = minibatch_size

    def get_next_minibatch(self):
        """
        yielding minibatches of the make_moons dataset
        :return: x and y of dataset for training
        """
        excerpt = None
        assert self.X_train.shape[0] == self.y_train.shape[0]
        indices = np.arange(self.X_train.shape[0])
        np.random.shuffle(indices)
        for start_index in range(0, self.X_train.shape[0] - self.minibatch_size + 1, self.minibatch_size):
            excerpt = indices[start_index:start_index + self.minibatch_size]
            if excerpt.any():
                yield self.X_train[excerpt], self.y_train[excerpt]