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


def lwlr_sine_pt(x_q, xm, ym, k):
    # convert to matrix
    xM = np.mat(xm)
    yM = np.mat(ym)
    # diagonal matrix
    m = np.shape(xM)[0]
    w = np.mat(np.eye((m)))
    # fill weights using Gaussian Kernel
    for i in range(m):
        diffM = x_q - xM[i, :]
        # print(np.shape(x_q))
        # print(diffM)
        w[i, i] = np.exp(diffM * diffM.T / (-2.0 * k**2))

    # reshape x_q and append 1
    x_q = np.reshape(x_q, (-1, 1))
    x_q = np.vstack((x_q, 1)).T

    # Find Beta
    xTwx = xM.T * w * xM
    B = xTwx.I * xM.T * w * yM
    # find and return x_q
    return x_q * B


def lwlr_sine(test, xm, ym, k):
    """ xm: nx(m+1) (col of 1s at end)
        ym: nxm or nx1 for singular
        Beta: (n+1)xm
        test: (n+1)xm (row of 1 at end)
    """
    m = np.shape(test)[0]
    y_hat = np.zeros(m)
    for i in range(m):
        # find Beta and hence y_hat for every x_q (test[i])
        y_hat[i] = lwlr_sine_pt(test[i], xm, ym, k)
        print("Completed {} of {}".format(i, m))
    return y_hat


def plot_sine(x, y, xt, yhat):
    # Sine Plot
    plt.figure(60)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Noisy Sine Wave LWLR Test')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.scatter(x, y, color='b', label="Training Set")
    plt.scatter(xt, yhat, color='r', label="Testing Set")
    plt.legend()
    plt.show()


def main():
    x, y = sine(2, 200, 0.05)
    x_test = np.linspace(0.1, 2, 300)
    # Reshaping below for matrix inv
    # convert into arrays
    xm = np.array(x)
    ym = np.array(y)
    ym = ym.flatten()
    # -1 indicates use input dimension
    ym = np.reshape(ym, (-1, 1))

    # test = np.append(x, 1)
    test = x_test
    test = np.reshape(test, (-1, 1))

    # -1 indicates use input dimension
    xm = np.reshape(xm, (-1, 1))
    ones_app = np.ones((np.shape(xm)[0], 1))
    xm = np.hstack((xm, ones_app))
    ym = ym

    k = 0.05
    # perform LWLR
    yhat = lwlr_sine(test, xm, ym, k)

    plot_sine(x, y, x_test, yhat)


if __name__ == "__main__":
    main()