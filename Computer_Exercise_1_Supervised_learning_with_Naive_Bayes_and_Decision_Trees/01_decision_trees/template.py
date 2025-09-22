# Author: Bjarki Þór Jónsson
# Date: 23 August 2025
# Project: 01 Decision Trees
# Acknowledgements: 
#


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''

    priors = np.zeros(len(classes))
    
    for i, cls in enumerate(classes):
        
        count = np.sum(targets == cls)
        
        priors[i] = count / len(targets)
    
    return priors


def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    mask = features[:, split_feature_index] < theta
    
    features_1 = features[mask == True]
    targets_1 = targets[mask == True]
    
    features_2 = features[mask == False]
    targets_2 = targets[mask == False]

    return (features_1, targets_1), (features_2, targets_2)

features, targets, classes = load_iris()
(f_1, t_1), (f_2, t_2) = split_data(features, targets, 2, 4.65)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    probs = prior(targets, classes)
    
    sum_of_squares = np.sum(probs ** 2)
    
    gini = 0.5 * (1 - sum_of_squares)
    
    return gini



def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = t1.shape[0] + t2.shape[0]
    return (t1.shape[0]*g1 + t2.shape[0]*g2) / n


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)
    
    impurity = weighted_impurity(t_1, t_2, classes)
    
    return impurity


def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None

    for i in range(features.shape[1]):

        thetas = np.linspace(features[:, i].min(), features[:, i].max(), num_tries + 2)[1:-1]

        for theta in thetas:

            gini = total_gini_impurity(features, targets, classes, i, theta)

            if gini < best_gini:
                best_gini = gini
                best_dim = i
                best_theta = theta

    return best_gini, best_dim, best_theta



class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        predictions = self.tree.predict(self.test_features)
        return np.mean(predictions == self.test_targets)

    def plot(self):
        feature_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
        class_names = ['Setosa', 'Versicolour', 'Virginica']
        plt.figure(figsize=(15, 10))
        plot_tree(self.tree, feature_names=feature_names, class_names=class_names, filled=True)
        plt.savefig('2_3_1.png')
        plt.show()

    def guess(self):
        class_names = ['Setosa', 'Versicolour', 'Virginica']
        predictions = self.tree.predict(self.test_features)
        return [class_names[prediction] for prediction in predictions]

    def confusion_matrix(self):
        predictions = self.tree.predict(self.test_features)
        matrix = np.zeros((3,3), dtype = int)
        for true, prediction in zip(self.test_targets, predictions):
            matrix[true, prediction] += 1
        return matrix


if __name__ == "__main__":
    features, targets, classes = load_iris()
    trainer = IrisTreeTrainer(features, targets, classes, train_ratio=0.8)
    trainer.train()
    print("Accuracy:", trainer.accuracy())
    print("Guesses:", trainer.guess()) 
    print("Confusion Matrix:\n", trainer.confusion_matrix()) 
    trainer.plot()

