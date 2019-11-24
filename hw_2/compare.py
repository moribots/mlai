#!/usr/bin/env python
# use chmod +x file_name.py (+x grants executable permission)
# use ./file_name.py to run
"""
Python 2.7 PEP8 Style
Code submission for Homework 2 Part A
Machine Learning & Artificial Intelligence for Robotics
Data Set 0
Locally Weighted Linear Regression for Motion Model Estimation

Data interpreted with Python's Pandas library

Maurice Rahme
Student ID: 3219435
mauricerahme2020@u.northwestern.edu
"""
from __future__ import division
import numpy as np
from random import randrange
import csv
import matplotlib.pyplot as plt


def lwlr_pt(x_q, xm, ym, k, xval, dimx, dimy):
    # convert to matrix
    # xM = np.mat(xm)
    # yM = np.mat(ym)
    xM = np.reshape(xm, (-1, dimx))
    yM = np.reshape(ym, (-1, dimy))
    # diagonal matrix
    m = np.shape(xM)[0]
    # w = np.mat(np.eye((m)))
    w = np.eye(m)
    # Set rejection threshold
    thresh = 0  # 0.012
    # fill weights using Gaussian Kernel
    for i in range(m):
        # diffM = x_q - xM[i, :]
        diffM = xM[i, :] - x_q
        dh = np.divide(np.dot(diffM.T, diffM), k)
        kernel = np.exp(-np.dot(dh, dh.T))
        w[i, i] = np.sqrt(kernel)
        if w[i, i] < thresh:
            w[i, i] = 0
        # print(diffM)
        # w[i, i] = np.exp(np.dot(diffM, diffM.T) / (-2.0 * k**2))
        # w[i, i] = np.sqrt(np.dot(diffM.T, diffM)) / k
    if xval is True:
        # Perform cross-validation by removing x_q weight from eqn
        i = 0
        for i in range(m):
            if w[i, i] == 1:
                # print(i)
                # print("ping")
                w[i, i] = 0
    # print("\n")

    # find Z, v
    Z = np.dot(w, xM)
    v = np.dot(w, yM)
    ZTZ = np.dot(Z.T, Z)

    # Find Beta
    # Try for inverse case
    try:
        # inverse
        inv = np.linalg.inv(ZTZ)
    except:
        # pseudoinverse
        print('pinv')
        inv = np.linalg.pinv(ZTZ)

    B = np.dot(np.dot(inv, Z.T), v)

    # find nLWR
    i = 0
    nLWR = 0
    for i in range(len(w)):
        nLWR += w[i, i]**2
    # find ri
    i = 0
    r = np.empty((len(xM), dimy))
    summ = 0
    for i in range(len(xM)):
        Zi = np.reshape(Z[i], (-1, 1))
        r[i] = np.dot(Zi.T, B) - v[i]
        summ += (r[i] / (1 - np.dot(np.dot(Z[i].T, inv), Z[i])))**2

    # calc xval mean sqrt err for xq
    MSE_q = (1 / nLWR) * summ

    # calc var for xq
    i = 0
    C_q = 0
    for i in range(len(r)):
        C_q += r[i]**2
    var_q = C_q / nLWR

    return np.dot(x_q.T, B), MSE_q, var_q


def lwlr(test, xm, ym, k):
    """ xm: nx(m+1) (col of 1s at end)
        ym: nxm or nx1 for singular
        Beta: (n+1)xm
        test: (n+1)xm (row of 1 at end)
    """
    dimy = np.shape(ym)[1]
    dimx = np.shape(xm)[1]
    m = np.shape(test)[0]
    y_hat = np.zeros((m, dimy))
    MSE = np.zeros((m, dimy))
    VAR = np.zeros((m, dimy))
    for i in range(m):
        # find Beta and hence y_hat for every x_q (test[i])
        y_hat[i], MSE[i], VAR[i] = lwlr_pt(test[i], xm, ym, k, False, dimx, dimy)
        print("Completed {} of {}".format(i, m))
    return y_hat, MSE, VAR


def train_test_split(dataset, split=0.90):
    train = np.empty((1, len(dataset[0])))
    train_size = split * len(dataset)
    test = dataset
    i = 0
    while len(train) < train_size:
        index = randrange(len(test))
        if i == 0:
            train = test[index, :]
            test = np.delete(test, index, axis=0)
        else:
            train = np.vstack((train, test[index, :]))
            test = np.delete(test, index, axis=0)
        i += 1
    return train, test


def cross_validation_split(dataset, folds=10):
    dataset_split = np.empty((1, len(dataset[0])))
    test = dataset
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = np.empty((1, len(dataset[0])))
        j = 0
        while len(fold) < fold_size:
            index = randrange(len(test))
            if j == 0:
                fold = test[index, :]
                test = np.delete(test, index, axis=0)
            else:
                fold = np.vstack((fold, test[index, :]))
                test = np.delete(test, index, axis=0)
            j += 1
        if i == 0:
            dataset_split = fold
        else:
            dataset_split = np.vstack((dataset_split, fold))
    return dataset_split


def lwlr_crossval(test, xm, ym, k):
    """ xm: nx(m+1) (col of 1s at end)
        ym: nxm or nx1 for singular
        Beta: (n+1)xm
        test: (n+1)xm (row of 1 at end)
    """
    dimy = np.shape(ym)[1]
    dimx = np.shape(xm)[1]
    m = np.shape(test)[0]
    y_hat = np.zeros((m, dimy))
    MSE = np.zeros((m, dimy))
    VAR = np.zeros((m, dimy))
    for i in range(m):
        # find Beta and hence y_hat for every x_q (test[i])
        y_hat[i], MSE[i], VAR[i] = lwlr_pt(test[i], xm, ym, k, True, dimx, dimy)
        if i % 200 == 0:
            print("Completed {} of {}".format(i, m))
    return y_hat, MSE, VAR


def setup(train, test, data):
    train = np.array(train)
    test = np.array(test)

    # Input (train, test)
    mes_head_train = train[:, 3]
    mes_head_test = test[:, 3]
    xval_head = data[:, 3]
    # Output (train)
    err_dist = train[:, 6]
    xval_dist = data[:, 6]

    # Now we need to format the data for LWLR
    # Reshape
    # Inputs
    mes_head_train = np.reshape(mes_head_train, (-1, 1))
    mes_head_test = np.reshape(mes_head_test, (-1, 1))
    xval_head = np.reshape(xval_head, (-1, 1))

    # Outputs
    ym_train = np.reshape(err_dist, (-1, 1))
    ym_xval = np.reshape(xval_dist, (-1, 1))

    # Now create ones col to append to inputs
    ones_col_train = np.ones((np.shape(mes_head_train)[0], 1))
    ones_col_test = np.ones((np.shape(mes_head_test)[0], 1))
    ones_col_xval = np.ones((np.shape(xval_head)[0], 1))

    xm_train = np.hstack((mes_head_train, ones_col_train))
    xm_test = np.hstack((mes_head_test, ones_col_test))
    xm_xval = np.hstack((xval_head, ones_col_xval))

    # print("xm_train shape: {}".format(np.shape(xm_train)))
    # print("xm_test shape: {}".format(np.shape(xm_test)))
    # print("ym_train shape: {}".format(np.shape(ym_train)))

    return xm_test, xm_train, ym_train, xm_xval, ym_xval


def plot(data, xm_test, y_hat, y_hat_x, MSE_x, VAR_x):
    data = np.array(data)
    # Train Input
    mes_head = data[:, 3]
    # Train Output
    err_dist = data[:, 6]

    # Test Input
    xm_test = xm_test[:, 3]

    print(np.shape(y_hat))
    print(np.shape(xm_test))

    plt.figure(1)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Measured Heading vs Distance Error')
    plt.ylabel('dist_err [m]')
    plt.xlabel('head [rad]')
    plt.scatter(mes_head, err_dist, color='b', label='Full Dataset')
    plt.scatter(xm_test, y_hat, color='r', label='LWLR Test')

    plt.figure(2)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Measured Heading vs Distance Error - LOOCV')
    plt.ylabel('dist_err [m]')
    plt.xlabel('head [rad]')
    plt.scatter(mes_head, err_dist, color='b', label='Full Dataset')
    plt.scatter(mes_head, y_hat_x, color='darkviolet', label='LOOCV')

    plt.figure(3)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Mean Squared Error for LOOCV')
    plt.ylabel('MSE Distance [m]')
    plt.xlabel('x_q element')
    plt.plot(range(len(MSE_x)), MSE_x, color='b', label="MSE")

    plt.figure(4)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Variance for LOOCV')
    plt.ylabel('VAR Distance [m]')
    plt.xlabel('x_q element')
    plt.plot(range(len(VAR_x)), VAR_x, color='b', label="Variance")

    plt.legend()
    plt.show()


def main():
    data = np.loadtxt(open("learning_dataset.csv"), delimiter=",")

    # Split data into testing and training sets
    train, test = train_test_split(data)

    # Setup data as inputs and outputs for testing and training
    # as well as cross-validation training
    xm_test, xm_train, ym_train, xm_xval, ym_xval = setup(train, test, data)
    k = 0.001
    # LWLR Fit
    y_hat, MSE, VAR = lwlr(xm_test, xm_train, ym_train, k)
    # LWLR XVal Test
    y_hat_x, MSE_x, VAR_x = lwlr_crossval(xm_xval, xm_xval, ym_xval, k)

    # Create CSV File for Fit
    with open("yhat.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(y_hat)

    with open("MSE.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(MSE)

    with open("VAR.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(VAR)

    # Create CSV File for Xval
    with open("yhat_xval.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(y_hat_x)

    with open("MSE_x.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(MSE_x)

    with open("VAR_x.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(VAR_x)

    plot(data, test, y_hat, y_hat_x, MSE_x, VAR_x)


if __name__ == "__main__":
    main()