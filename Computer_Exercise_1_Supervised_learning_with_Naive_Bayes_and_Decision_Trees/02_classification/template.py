# Author: Bjarki Þór Jónsson
# Date: 26. Ágúst
# Project: 02 Classification
# Acknowledgements: 
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def gen_data(
    n: int,
    locs: np.ndarray,
    scales: np.ndarray
) -> np.ndarray:
    '''
    Return n data points, their classes and a unique list of all classes, from each normal distributions
    shifted and scaled by the values in locs and scales
    '''
    feature_list = []
    target_list = []
    classes = list(range(len(locs)))

    for i in classes:
        normdist = norm(loc=locs[i], scale=scales[i])

        points = normdist.rvs(n)

        feature_list.append(points)
        target_list.extend([i] * n)

    features = np.concatenate(feature_list)
    targets = np.array(target_list)

    return features, targets, classes



def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    class_features = features[targets == selected_class]
    
    class_mean = np.mean(class_features)
    
    return class_mean


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    class_feature = features[targets == selected_class]

    class_cov = np.var(class_feature, ddof=1)

    return class_cov


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    std_dev= np.sqrt(class_covar)

    return norm.pdf(feature, class_mean, std_dev)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        mean = mean_of_class(train_features, train_targets, class_label)
        means.append(mean)

        cov = covar_of_class(train_features, train_targets, class_label)
        covs.append(cov)
    likelihoods = []
    for i in range(test_features.shape[0]):
        point_likelihoods = []
        for k in range(len(classes)):
            likelihood = likelihood_of_class(test_features[i], means[k], covs[k])
            point_likelihoods.append(likelihood)
        likelihoods.append(point_likelihoods)
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    likelihood = np.argmax(likelihoods, axis=1)
    
    return likelihood


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """


    
    features, targets, classes = gen_data(25, np.array([-1, 1]), np.array([np.sqrt(5), np.sqrt(5)]))
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.8)
    


    '''
    # Section 2:
    plt.figure()
    
    # Get class 0 points
    class0_features = features[targets==0]
    
    # Plot class 0
    plt.scatter(
        class0_features,
        np.zeros(len(class0_features)),
        c='blue',
        marker='o',
    )
    
    # Get class 1 points
    class1_features = features[targets==1]
    
    # Plot class 1
    plt.scatter(
        class1_features,
        np.zeros(len(class1_features)),
        c='red',
        marker='s',
    )
    
    plt.savefig('2_1.png')
    
    plt.close()
    '''

    '''
    # Test Section 3
    print("Mean of class 0:", mean_of_class(train_features, train_targets, 0))
    print("Mean of class 1:", mean_of_class(train_features, train_targets, 1))
    '''

    #'''
    # Test Section 4
    print("Variance of class 0:", covar_of_class(train_features, train_targets, 0))
    print("Variance of class 1:", covar_of_class(train_features, train_targets, 1))
    #'''

    '''
    # Test section 5
    class_mean = mean_of_class(train_features, train_targets, 0)
    class_cov = covar_of_class(train_features, train_targets, 0)
    print("Likelihoods for class 0 (first 3 test points):", likelihood_of_class(test_features[0:3], class_mean, class_cov))
    class_mean = mean_of_class(train_features, train_targets, 1)
    class_cov = covar_of_class(train_features, train_targets, 1)
    print("Likelihoods for class 1 (first 3 test points):", likelihood_of_class(test_features[0:3], class_mean, class_cov))
    '''

    '''
    # Test Section 6
    means = np.array([mean_of_class(train_features, train_targets, 0), mean_of_class(train_features, train_targets, 1)])
    covs = np.array([covar_of_class(train_features, train_targets, 0), covar_of_class(train_features, train_targets, 1)])
    print("Likelihoods for test points:\n", maximum_likelihood(train_features, train_targets, test_features, classes))
    

    # Test Section 7
    likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    print("Predictions for test points:", predict(likelihoods))
    '''

    '''
    # Section 8: New dataset

    # Original dataset
    features, targets, classes = gen_data(25, np.array([-1, 1]), np.array([np.sqrt(5), np.sqrt(5)]))
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.8)
    likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    predictions = predict(likelihoods)
    accuracy_original = np.mean(predictions == test_targets)
    print("Accuracy for original dataset:", accuracy_original)

    # New dataset
    features, targets, classes = gen_data(25, np.array([-4, 4]), np.array([np.sqrt(2), np.sqrt(2)]))
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.8)
    likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    predictions = predict(likelihoods)
    accuracy_new = np.mean(predictions == test_targets)
    print("Accuracy for new dataset:", accuracy_new)
    
    '''
    
    pass
