from typing import Union
import numpy as np

from tools import load_iris, split_train_test


def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    if np.isscalar(x):
        if x < -100:
            return 0
        else:
            return 1 / (1 + np.exp(-x))
    else:
        mask = x < -100
        result = 1 / (1 + np.exp(-x))
        result[mask] = 0
        return result


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    sig = sigmoid(x)
    return sig * (1 - sig)


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    weight_sum = np.dot(x,w)
    sig_weight_sum = sigmoid(weight_sum)
    return weight_sum, sig_weight_sum


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    z0 = np.insert(x, 0, 1)

    a1 = np.dot(z0, W1)

    hidden_outputs = sigmoid(a1)
    z1 = np.insert(hidden_outputs, 0, 1)

    a2 = np.dot(z1, W2)

    y = sigmoid(a2)

    return y, z0, z1, a1, a2


def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    ...


def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    ...


def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    ...


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """

    #print(sigmoid(0.5))  # Should be about 0.6224593312018546
    #print(d_sigmoid(0.2))  # Should be about 0.24751657271185995

    p1 = perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1]))
    p2 = perceptron(np.array([0.2,0.4]),np.array([0.1,0.4]))

    #print(p1)
    #print(p2)


    np.random.seed(1234)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = \
        split_train_test(features, targets)
    
    np.random.seed(1234)

    # Take one point:
    x = train_features[0, :]
    K = 3 # number of classes
    M = 10
    D = 4
    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

    print("y:", y)
    print("z0:", z0)
    print("z1:", z1)
    print("a1:", a1)
    print("a2:", a2)
    
    pass