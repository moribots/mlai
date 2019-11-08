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


def sine(cycles, points, var):
    T = 1  # period
    y_var = var  # variance
    x = []
    y = []
    for t in np.linspace(0, cycles * T, points):
        x_pt = t
        x.append(x_pt)
        y_pt = np.sin((2 * np.pi * t) / T) + y_var * np.random.randn(1)
        y.append(y_pt)

    return x, y


def lwlr_pt(x_q, xm, ym, k):
    # convert to matrix
    xM = np.mat(xm)
    yM = np.mat(ym)
    # diagonal matrix
    m = np.shape(xM)[0]
    weights = np.mat(np.eye((m)))
    # pop weights with exp decay vals
    # using Gaussian Kernel
    for i in range(m):
        diffM = x_q - xM[i, :]
        weights[i, i] = np.exp(diffM * diffM.T / (-2.0 * k**2))

    # find x_q
    xTwx = xM.T * (weights * xM)
    # if np.linalg.det(xTwx) == 0.0:
    #     print("This matrix is singular, cannot do inverse")
    #     return
    x_q = np.reshape(x_q, (-1, 1))
    x_q = np.vstack((x_q, 1)).T

    ws = xTwx.I * (xM.T * (weights * yM))
    return x_q * ws


def lwlr(train, xm, ym, k):
    """ Xarr: nx(m+1) (col of 1s at end)
        Yarr: nxm or nx1 for singular
        Beta: (n+1)xm
        trainArr: (n+1)xm (row of 1 at end)
    """
    m = np.shape(train)[0]
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = lwlr_pt(train[i], xm, ym, k)
        # ws.append(y_hat[i] / trainArr[i])
        print("Completed {} of {}".format(i, m))
    return y_hat


def plot(x, y, yhat):
    # Sine Plot
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Noisy Sine Wave')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.scatter(x, y)
    plt.scatter(x, yhat, color='r')

    plt.show()


def main():
    x, y = sine(2, 500, 0.05)
    x2, y2 = sine(2, 500, 0)
    # Reshaping below for matrix inv
    # convert into arrays
    xm = np.array(x)
    ym = np.array(y)
    ym = ym.flatten()
    # -1 indicates use input dimension
    ym = np.reshape(ym, (-1, 1))

    # train = np.append(x, 1)
    train = x
    train = np.reshape(train, (-1, 1))

    # -1 indicates use input dimension
    xm = np.reshape(xm, (-1, 1))
    ones_app = np.ones((np.shape(xm)[0], 1))
    xm = np.hstack((xm, ones_app))
    ym = ym

    k = 0.05
    # perform LWLR
    yhat = lwlr(train, xm, ym, k)

    plot(x, y, yhat)


if __name__ == "__main__":
    main()