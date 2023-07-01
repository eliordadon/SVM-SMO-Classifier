"""
utils.py this file contains various helper functions
author: Elior Dadon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # for plt.show()
from sklearn.metrics import confusion_matrix  # for calculating the confusion matrix
from sklearn.metrics import classification_report # calculate the table of confusion and the sensitivity


def linear_kernel(x_i, x_j):
    """
    :param x_i: first numpy array
    :param x_j: second numpy array
    :return: linear kernel between x_i and x_j (float)
    """
    return np.dot(x_i, x_j.T)


def rbf_kernel(x_i, x_j, gamma=1.0):
    """
    :param x_i: first numpy array
    :param x_j: second numpy array
    :param gamma: kernel coefficient, default is 1.0
    :return: RBF kernel between x_i and x_j (float)
    """
    return np.exp(-gamma * np.linalg.norm(x_i - x_j) ** 2)


def get_X_y(dataset):
    """
    Return the data and target from a sklearn dataset
    """
    return dataset.data, dataset.target


def iris_scatter_plot_matrix(iris):
    """
    Generates a scatter plot matrix for the Iris dataset.

    :param iris: Dictionary containing the Iris dataset, including 'data', 'target', and 'feature_names'.
    :return: none
    """
    iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                           columns=iris['feature_names'] + ['species'])

    pd.plotting.scatter_matrix(iris_df, figsize=(10, 10), diagonal='hist', c=iris_df['species'])
    plt.show()


def model_evaluation(y_test, predictions, kernel_name):
    """
    evaluate the svm model with confusion matrix, accuracy and table of confusion
    :param y_test: real target values
    :param predictions: model predictions
    :param kernel_name: the kernel name that was used in the svm model
    :return: none
    """

    confusion_mat = confusion_matrix(y_test, predictions)
    accuracy = sum(predictions == y_test) / len(y_test)
    classification_rep = classification_report(y_test, predictions, zero_division=0)

    print("\n" + kernel_name + " Confusion Matrix:")
    print(confusion_mat)
    print("\n" + kernel_name + " Accuracy: {:.2f}%".format(accuracy * 100))
    print("\n" + kernel_name + " Classification Report:")
    print(classification_rep)



