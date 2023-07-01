"""
main.py this file contains the main function which test the svm implementation with the Iris dataset (you can change it)
author: Elior Dadon
"""

import numpy as np
from sklearn import datasets  # for Iris dataset
from sklearn.model_selection import train_test_split  # for splitting the data
from utils import linear_kernel, rbf_kernel, iris_scatter_plot_matrix, get_X_y,  model_evaluation
from svm_smo import train_svm, multiclass_predict


def main():
    # Load the Iris dataset
    iris = datasets.load_iris()
    # Scatter plot of the Iris dataset
    iris_scatter_plot_matrix(iris)

    X, y = get_X_y(iris)

    # split the data into train, validation and test set (60/20/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

    # Dictionary to store all of the binary SVMs (storing the trained models for each class, where the key is the class
    # label and the value is the trained model) (we also making one for the rbf type, the default one is linear kernel)
    binary_svms = {}
    binary_svms_rbf = {}

    # Iterate over all the classes
    for label in np.unique(y):
        # Create a binary target variable where the class is considered the positive class
        binary_labels = np.where(y == label, 1, -1)

        # Train the binary SVM
        alpha, b = train_svm(X_train, binary_labels)
        alpha_rbf, b_rbf = train_svm(X_train, binary_labels, kernel_type='rbf')

        # Store the SVM in the dictionary
        binary_svms[label] = (alpha, b)
        binary_svms_rbf[label] = (alpha_rbf, b_rbf)

    # predict the test set (Linear Kernel)
    predictions = [multiclass_predict(x, X_train, y_train, binary_svms, kernel=linear_kernel) for x in X_test]
    # evaluate model with linear kernel
    model_evaluation(y_test, predictions, "Linear Kernel")

    # predict the test set (RBF Kernel)
    predictions_rbf = [multiclass_predict(x, X_train, y_train, binary_svms_rbf, kernel=rbf_kernel) for x in X_test]
    # evaluate model with linear kernel
    model_evaluation(y_test, predictions, "RBF Kernel")


if __name__ == '__main__':
    main()
