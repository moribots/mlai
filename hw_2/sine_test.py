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


def sine(cycles, points):
    T = 1  # period
    y_var = 0.05  # variance
    x = []
    y = []
    for t in np.linspace(0, cycles * T, points):
        x_pt = t
        x.append(x_pt)
        y_pt = np.sin((2 * np.pi * t) / T) + y_var * np.random.randn(1)
        y.append(y_pt)

    return x, y


def lwlr_pt(testPoint, xArr, yArr, k):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # Create diagonal matrix
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    # Populate weights with exponentially decaying values
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k**2))

    # find extimate for testpoint
    xTwx = xMat.T * (weights * xMat)
    # if np.linalg.det(xTwx) == 0.0:
    #     print("This matrix is singular, cannot do inverse")
    #     return

    ws = xTwx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlr(testArr, xArr, yArr, k):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    # Replace each entry by its weight
    for i in range(m):
        yHat[i] = lwlr_pt(testArr[i], xArr, yArr, k)
    return yHat


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
    x, y = sine(2, 500)
    # convert into arrays
    x = np.array(x)
    y = np.array(y)
    y = y.flatten()

    # perform LWLR
    k = 0.1
    yhat = lwlr(x[:250], x[251:], y[251:], k)
    plot(x, y, yhat)


if __name__ == "__main__":
    main()