"""
svm_smo.py this file contains implementation of an SVM classifier using The SMO algorithm from the paper
"Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines, 1998"
author: Elior Dadon
"""

import numpy as np
from random import randint
from sklearn.model_selection import train_test_split  # for splitting the data
from sklearn.datasets import make_classification  # for dummy dataset
from sklearn.metrics import accuracy_score  # to calculate model accuracy (used in this file for the test_smo() function)
from utils import linear_kernel, rbf_kernel


def predict(x, alpha, b, point, target, kernel=linear_kernel):
    """
    This function makes a prediction on a single example x using the learned SVM parameters alpha, b,
    and the training data point, target.
    :return: The predicted label for x (-1 / 1), the raw score (without the sign)
    """
    # Initialize the prediction to the bias term
    pred = b

    # Loop over all the training examples
    for i in range(len(target)):
        # If the alpha for this example is not zero, add its contribution to the prediction
        if alpha[i] > 0:  # support vectors
            pred += alpha[i] * target[i] * kernel(point[i], x)

    label_pred = pred
    if np.sign(pred) == 0:  # np.sign will return 0 if the value is 0
        label_pred = 1
    # Return the sign of the prediction to get the final label
    return label_pred, pred


def predict_set(X, alpha, b, point, target, kernel):
    """
    This function makes a prediction on a whole data set X using the learned SVM parameters alpha, b,
    and the training data point, target.
    :return: list of predictions for each x in X
    """
    set_pred = []
    for x in X:
        label_pred, _ = predict(x, alpha, b, point, target, kernel)
        set_pred.append(label_pred)

    return set_pred


def multiclass_predict(x, point, target, models, kernel):
    """
    This function makes a prediction on a single example x using the learned SVM parameters
    for each class, and return the class that has the highest prediction score
    :return: the best class
    """
    # Initialize the best class to -1 and the best score to negative infinity
    best_class = -1
    best_score = -np.inf

    # Iterate over all the binary classifiers
    for label, (alpha, b) in models.items():
        # Make a prediction using the current SVM
        score = predict(x, alpha, b, point, target, kernel)[1]

        # Update the best class and best score if this prediction is better than the previous best
        if score > best_score:
            best_score = score
            best_class = label

    return best_class


def objective_function(a, i1, i2, alpha, target, point, b, error_cache, kernel):
    """
    Calculates the objective function used in the optimization process.

    :param a: A scalar parameter that can be either H or L.
    :param i1: Index representing the position of the first data point.
    :param i2: Index representing the position of the second data point.
    :param alpha: Array of coefficients corresponding to the support vectors.
    :param target: Array containing the target labels of the data points.
    :param point: Array of feature vectors representing the data points.
    :param b: Bias term.
    :param error_cache: Array storing the error values associated with each data point.
    :param kernel: Function that calculates the kernel value between two data points.
    :return: The value of the objective function.
    """
    y1 = target[i1]
    y2 = target[i2]
    x1 = point[i1]
    x2 = point[i2]
    alpha1 = alpha[i1]
    alpha2 = alpha[i2]
    E1 = error_cache[i1]
    E2 = error_cache[i2]
    s = y1 * y2

    # equations in (19) in the paper
    f1 = y1 * (E1 + b) - alpha1 * kernel(x1, x1) - s * alpha2 * kernel(x1, x2)
    f2 = y2 * (E2 + b) - alpha2 * kernel(x2, x2) - s * alpha1 * kernel(x1, x2)
    LH1 = alpha1 + s * (alpha2 - a)  # a = H or L

    return LH1 * f1 + a * f2 + 0.5 * LH1 ** 2 * kernel(x1, x1) + 0.5 * a ** 2 * kernel(x2, x2) + s * a * LH1 * kernel(
        x1, x2)


def update_error_cache(i, error_cache, alpha, target, point, b, kernel):
    """
    Update the error cache for the given example (point[i])
    :return: None
    """
    # predict the class label of example i
    score = predict(point[i], alpha, b, point, target, kernel)[1]  # raw score without sign
    # calculate the error
    error = score - target[i]
    # update the error cache
    error_cache[i] = error


def second_choice_heuristic(i2, target, alpha, error_cache, C):
    """
    This function implements the second choice heuristic for selecting the second variable (i1) in the optimization problem,
    as described in the paper in section 2.2.
    :return: the second variable index (i1)
    """
    max_diff = -1
    i1 = -1
    E2 = error_cache[i2]
    non_bound_alphas = np.where((alpha > 0) & (alpha < C))[0]
    for i in non_bound_alphas:
        E1 = error_cache[i]
        if abs(E1 - E2) > max_diff:
            i1 = i
            max_diff = abs(E1 - E2)
    if i1 == -1:
        i1 = i2
        while i1 == i2:
            i1 = randint(0, len(target) - 1)
    return i1


def takeStep(i1, i2, target, point, alpha, b, error_cache, C, kernel, eps=1e-3):
    """
    Attempts to update the Lagrange multipliers alpha[i1] and alpha[i2]
    for the training points point[i1] and point[i2].
    :param i1: Index of the first training example (int).
    :param i2: Index of the second training example (int).
    :param target: Vector of class labels for the training examples (np.ndarray).
    :param point: Matrix of training examples (np.ndarray).
    :param alpha: Vector of Lagrange multipliers (np.ndarray).
    :param b: Threshold for the decision boundary (float).
    :param error_cache: Vector of error values for the training examples (np.ndarray).
    :param C: Regularization constant (float).
    :param eps: Tolerance value used as a stopping criterion for optimization (float).
    :param kernel: kernel function.
    :return: Returns 1 if the multipliers are successfully updated, and 0 otherwise.
    """

    # if the chosen alphas are the same, skip this step
    if i1 == i2:
        return 0
    alpha1 = alpha[i1]  # lagrange multiplier for i1
    alpha2 = alpha[i2]
    y1 = target[i1]
    y2 = target[i2]
    E1 = error_cache[i1]  # = SVM output on point[i1] - y1
    E2 = error_cache[i2]
    s = y1 * y2

    # Compute L, H via equations (13) and (14) in the paper
    if y1 == y2:
        L = max(0, alpha1 + alpha2 - C)
        H = min(C, alpha1 + alpha2)
    else:
        L = max(0, alpha2 - alpha1)
        H = min(C, C + alpha2 - alpha1)
    if L == H:
        return 0

    # compute eta
    # eta represents the change in the objective function if the lagrange multipliers alpha1 and alpha2 are updated
    k11 = kernel(point[i1], point[i1])
    k12 = kernel(point[i1], point[i2])
    k22 = kernel(point[i2], point[i2])
    eta = k11 + k22 - 2 * k12
    if eta > 0:  # examples are similar (according to the kernel function)
        a2_new = alpha2 + y2 * (E1 - E2) / eta
        if a2_new < L:
            a2_new_clipped = L
        elif a2_new > H:
            a2_new_clipped = H
        else:
            a2_new_clipped = alpha2
    else:  # examples dissimilar
        Lobj = objective_function(L, i1, i2, alpha, target, point, b, error_cache, kernel)
        Hobj = objective_function(H, i1, i2, alpha, target, point, b, error_cache, kernel)
        if Lobj < Hobj - eps:
            a2_new_clipped = L
        elif Lobj > Hobj + eps:
            a2_new_clipped = H
        else:
            a2_new_clipped = alpha2
    if abs(a2_new_clipped - alpha2) < eps * (a2_new_clipped + alpha2 + eps):
        return 0
    a1_new = alpha1 + s * (alpha2 - a2_new_clipped)

    # Update threshold to reflect change in Lagrange multipliers
    b1 = b - E1 - y1 * (a1_new - alpha1) * k11 - y2 * (a2_new_clipped - alpha2) * k12
    b2 = b - E2 - y1 * (a1_new - alpha1) * k12 - y2 * (a2_new_clipped - alpha2) * k22
    if 0 < a1_new < C:
        b = b1
    elif 0 < a2_new_clipped < C:
        b = b2
    else:
        b = (b1 + b2) / 2

    # Update weight vector to reflect change in a1_new & a2_new, if SVM is linear
    # update_weight_vector(i1, i2, a1_new, a2_new_clipped, target, point)

    # Store a1_new in the alpha array
    alpha[i1] = a1_new
    # Store a2_new_clipped in the alpha array
    alpha[i2] = a2_new_clipped

    # Update error cache using new Lagrange multipliers
    update_error_cache(i1, error_cache, alpha, target, point, b, kernel)
    update_error_cache(i2, error_cache, alpha, target, point, b, kernel)

    return 1


def examineExample(i2, target, point, alpha, b, error_cache, C, tol, kernel):
    """
    Examines the training point point[i2] to determine whether it is a
    support vector and whether the Lagrange multiplier alpha[i2] should be updated.
    :param i2: the index of the example being examined (int)
    :param target: Vector of class labels for the training examples (np.ndarray).
    :param point: Matrix of training examples (np.ndarray).
    :param alpha: Vector of Lagrange multipliers (np.ndarray).
    :param b: Threshold for the decision boundary (float).
    :param error_cache: Vector of error values for the training examples (np.ndarray).
    :param C: Regularization constant (float).
    :param tol: a tolerance value for the KKT conditions, which is used as a stopping criteria for the optimization process (float).
    :param kernel: kernel function
    :return: Returns 1 if the multiplier is updated, and 0 otherwise.
    """
    y2 = target[i2]
    alpha2 = alpha[i2]
    E2 = error_cache[i2]
    r2 = E2 * y2
    if (r2 < -tol and alpha2 < C) or (r2 > tol and alpha2 > 0):
        # Number of non-zero and non-C alpha
        non_bound_alphas = [a for a in alpha if 0 < a < C]
        if len(non_bound_alphas) > 1:
            # Result of second choice heuristic
            i1 = second_choice_heuristic(i2, target, alpha, error_cache, C)
            if takeStep(i1, i2, target, point, alpha, b, error_cache, C, kernel):
                return 1

        for i1 in non_bound_alphas:
            if takeStep(i1, i2, target, point, alpha, b, error_cache, C, kernel):
                return 1
        for i1 in range(len(alpha)):
            if takeStep(i1, i2, target, point, alpha, b, error_cache, C, kernel):
                return 1
    return 0


def train_svm(point, target, C=1.0, tol=1e-3, kernel_type='linear'):
    """
    Trains an SVM using the SMO algorithm.
    :param target: Vector of class labels for the training examples (np.ndarray).
    :param point: Matrix of training examples (np.ndarray).
    :param C: Regularization constant, default is 1.0 (float)
    :param tol: a tolerance value for the algorithm, which controls the stopping criteria for the optimization process (float).
    :param kernel_type: type of the kernel (linear / rbf ...) default is linear (string).
    :return: Returns the Lagrange multipliers alpha and the threshold b.
    """
    num_changed = 0
    examine_all = 1
    alpha = np.zeros(len(target))
    b = 0
    error_cache = np.zeros(len(target))
    kernels = {'linear': linear_kernel, 'rbf': rbf_kernel}
    kernel = kernels[kernel_type]
    while num_changed > 0 or examine_all:
        num_changed = 0
        if examine_all:
            # Loop over all training examples
            for i in range(len(target)):
                num_changed += examineExample(i, target, point, alpha, b, error_cache, C, tol, kernel)
        else:
            # Loop over examples where alpha is not 0 & not C
            non_bound_alphas = [i for i in range(len(target)) if 0 < alpha[i] < C]
            for i in non_bound_alphas:
                num_changed += examineExample(i, target, point, alpha, b, error_cache, C, tol, kernel)
        if examine_all:
            examine_all = 0
        elif num_changed == 0:
            examine_all = 1
    return alpha, b


def test_smo():
    """
    test the smo svm classifier on a dummy dataset
    :return: None
    """
    # Create dummy dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, n_informative=5, n_clusters_per_class=2,
                               class_sep=2.0, flip_y=0.1, random_state=42)
    y = 2 * y - 1  # Changing labels to 1 and -1

    # Split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the SMO SVM classifier
    alpha, b = train_svm(X_train, y_train)

    # Predict the labels for the test set
    y_pred = predict_set(X_test, alpha, b, X_train, y_train, kernel=linear_kernel)

    # Compute the accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc:.2f}')