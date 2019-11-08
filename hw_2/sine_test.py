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


def plot(x, y):
    # Sine Plot
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Noisy Sine Wave')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.scatter(x, y)

    plt.show()



def main():
    x, y = sine(2, 500)
    plot(x, y)


if __name__ == "__main__":
    main()