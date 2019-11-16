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
import matplotlib.pyplot as plt


def sine(cycles, pts, var):
    T = 1  # period
    y_var = var  # variance
    x = []
    y = []
    for t in np.linspace(0, cycles * T, pts):
        y_pt = np.sin((2 * np.pi * t) / T) + y_var * np.random.randn(1)
        y.append(y_pt)
        x.append(t)
    return x, y


def lwlr_pt(x_q, xm, ym, k):

    # convert to matrix
    xM = np.mat(xm)
    yM = np.mat(ym)
    # diagonal matrix
    m = np.shape(xM)[0]
    w = np.mat(np.eye((m)))
    # fill weights using Gaussian Kernel
    for i in range(m):
        diffM = x_q - xM[i, :]
        # print(x_q)
        # print(xM[i, :])
        # print(diffM)
        # print("\n")
        w[i, i] = np.exp(diffM * diffM.T / (-2.0 * k**2))

    # Find Beta
    xTwx = xM.T * w * xM
    # Try for inverse case
    try:
        # inverse
        inv = np.linalg.inv(xTwx)
    except:
        # pseudoinverse
        print('pinv')
        inv = np.linalg.pinv(xTwx)

    B = inv * xM.T * w * yM
    # print(B)
    # print(x_q)

    # print(x_q * B)
    # find and return x_q
    return x_q * B


def lwlr(test, xm, ym, k):
    """ xm: nx(m+1) (col of 1s at end)
        ym: nxm or nx1 for singular
        Beta: (n+1)xm
        test: (n+1)xm (row of 1 at end)
    """
    m = np.shape(test)[0]
    y_hat = np.zeros((m, 2))
    # print(y_hat[0, 0])

    # print(test[0])

    for i in range(m):
        # find Beta and hence y_hat for every x_q (test[i])
        y_hat[i] = lwlr_pt(test[i], xm, ym, k)
        print("Completed {} of {}".format(i, m))
    return y_hat


def plot(x, y, xt, yhat):
    # Sine Plot
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Noisy Sine Wave')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.scatter(x, y, color='b')
    plt.scatter(xt, yhat, color='r')

    plt.show()


def main():
    x, y = sine(2, 300, 0.05)
    x2, y2 = sine(2, 300, 0.05)
    x_test = np.linspace(0.1, 2, 200)
    x2_test = np.linspace(0.2, 2, 200)
    # Reshaping below for matrix inv
    # convert into arrays
    xm = np.array(x)
    ym = np.array(y)
    ym = ym.flatten()
    x2m = np.array(x2)
    y2m = np.array(y2)
    y2m = y2m.flatten()
    # -1 indicates use input dimension
    xm = np.reshape(xm, (-1, 1))
    x2m = np.reshape(x2m, (-1, 1))
    ym = np.reshape(ym, (-1, 1))
    y2m = np.reshape(y2m, (-1, 1))

    # append 2nd col for 2nd sine
    xm = np.hstack((xm, x2m))
    ym = np.hstack((ym, y2m))

    # test = np.append(x, 1)
    test = x_test
    test = np.reshape(test, (-1, 1))
    x2_test = np.reshape(x2_test, (-1, 1))
    test = np.hstack((test, x2_test))
    test1 = test
    # print(np.shape(test))

    # ones_row = np.ones((1, np.shape(test)[1]))
    # test = np.vstack((test, ones_row))

    ones_col = np.ones((np.shape(xm)[0], 1))
    xm = np.hstack((xm, ones_col))
    ones_col = np.ones((np.shape(test)[0], 1))
    test = np.hstack((test, ones_col))
    ym = ym

    k = 0.05
    # perform LWLR
    yhat = lwlr(test, xm, ym, k)

    yhat1, yhat2 = np.hsplit(yhat, 2)
    test2, test3 = np.hsplit(test1, 2)

    print(np.shape(yhat1))

    plot(x, y, test1, yhat)


if __name__ == "__main__":
    main()