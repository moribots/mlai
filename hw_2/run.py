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


def viz_data(gt_dead, gt_train, odom_train):
    """ Plot error in data


        NOTE:

        input [v w]^T * dt
        output [dx dy dtheta] (all ground truth)

        also assuming no prior controls and that controls
        extend to next gt_train
    """
    # start by turning data into 2D array
    gt_dead = np.array(gt_dead)
    gt_train = np.array(gt_train)
    odom_dt = np.array(odom_train)

    n = np.shape(odom_train)[0]

    for i in range(n - 1):
        odom_dt[i, 1:] = odom_dt[i, 1:] * (gt_train[i + 1, 0] - gt_train[i, 0])

    dmag_x = gt_train[:, 1]
    dmag_y = gt_train[:, 2]
    dmag_h = gt_train[:, 3]
    dmag_gt_train = np.sqrt(np.square(dmag_x) + np.square(dmag_y))

    # now calculate x difference
    x_diff = gt_dead[:, 1] - dmag_x
    y_diff = gt_dead[:, 2] - dmag_y

    # now calculate distance magnitude difference
    diff_dmag = np.sqrt(np.square(x_diff) + np.square(y_diff))

    # now calculate heading difference for each set
    diff_head = gt_dead[:, 3] - dmag_h

    return diff_dmag, diff_head, dmag_gt_train, dmag_x, dmag_y, dmag_h, odom_dt


def plot_viz(odom, diff_dmag, diff_head, dmag_gt, dmag_x, dmag_y, dmag_h, odom_dt):
    """
    """

    # First turn odom into array
    odom = np.array(odom)

    # remove first datapoint from diff_dmag and diff_head
    diff_dmag = np.delete(diff_dmag, (0), axis=0)
    diff_head = np.delete(diff_head, (0), axis=0)
    dmag_gt = np.delete(dmag_gt, (0), axis=0)
    dmag_x = np.delete(dmag_x, (0), axis=0)
    dmag_y = np.delete(dmag_y, (0), axis=0)
    dmag_h = np.delete(dmag_h, (0), axis=0)

    # Initialize Plot dmagv dt
    plt.figure(1)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance error for v commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('vdt [m]')
    plt.scatter(odom_dt[:, 1], diff_dmag)

    # Initialize Plot dmagv
    plt.figure(2)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance error for v commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('vdt [m/s]')
    plt.scatter(odom[:, 1], diff_dmag)

    # Initialize Plot dmagw dt
    plt.figure(3)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance error for w commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('wdt [rad]')
    plt.scatter(odom_dt[:, 2], diff_dmag)

    # Initialize Plot dmagw
    plt.figure(4)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance error for w commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('wdt [rad/s]')
    plt.scatter(odom[:, 2], diff_dmag)

    # Initialize Plot dmagv dt
    plt.figure(5)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance change for v commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('vdt [m]')
    plt.scatter(odom_dt[:, 1], dmag_gt)

    # Initialize Plot dmagv
    plt.figure(6)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance change for v commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('vdt [m/s]')
    plt.scatter(odom[:, 1], dmag_gt)

    # Initialize Plot dmagw dt
    plt.figure(7)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance change for w commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('wdt [rad]')
    plt.scatter(odom_dt[:, 2], dmag_gt)

    # Initialize Plot dmagw
    plt.figure(8)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance change for w commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('wdt [rad/s]')
    plt.scatter(odom[:, 2], dmag_gt)

    plt.show()


def plot(x, y, xt, yhat):
    # Sine Plot
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Noisy Sine Wave')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.scatter(x, y, color='b')
    plt.scatter(xt, yhat, color='r')

    plt.show()


def setup(dmag_gt, dmag_h, dmag_x, dmag_y, odom_train, odom_dt):

    # First turn odom into array
    odom_train = np.array(odom_train)
    odom_dt = np.array(odom_dt)

    # remove first datapoint
    dmag_gt = np.delete(dmag_gt, (0), axis=0)
    dmag_x = np.delete(dmag_x, (0), axis=0)
    dmag_y = np.delete(dmag_y, (0), axis=0)
    dmag_h = np.delete(dmag_h, (0), axis=0)

    # Extract commands (independent vars)
    v = odom_train[:, 1]
    w = odom_train[:, 2]

    vdt = odom_dt[:, 1]
    wdt = odom_dt[:, 2]

    # Reshape
    v = np.reshape(v, (-1, 1))
    vdt = np.reshape(vdt, (-1, 1))

    w = np.reshape(w, (-1, 1))
    wdt = np.reshape(wdt, (-1, 1))

    dmag_gt = np.reshape(dmag_gt, (-1, 1))
    dmag_x = np.reshape(dmag_gt, (-1, 1))
    dmag_y = np.reshape(dmag_gt, (-1, 1))
    dmag_h = np.reshape(dmag_gt, (-1, 1))

    # Now create xm (input matrix)
    xm = np.hstack((v, w))
    xmdt = np.hstack((vdt, wdt))

    # Now create ym (output matrix)
    ymabs = np.hstack((dmag_gt, dmag_h))
    ymcart = np.hstack((dmag_x, dmag_y, dmag_h))

    # TODO:
    # Add test input here, for now use train

    # Add ones col at end of inputs
    ones_col = np.ones((np.shape(xm)[0], 1))
    ones_coldt = np.ones((np.shape(xmdt)[0], 1))
    xm = np.hstack((xm, ones_col))
    xmdt = np.hstack((xmdt, ones_coldt))

    # Now limit to number of points (lest lwlr take too long)
    num = 1000
    xm = xm[:num, :]
    xmdt = xmdt[:num, :]
    ymabs = ymabs[:num, :]
    ymcart = ymcart[:num, :]

    return xm, xmdt, ymabs, ymcart

def main():
    gt_train = np.loadtxt(open("gt_train.csv"), delimiter=",")
    gt_dead = np.loadtxt(open("gt_deadreck.csv"), delimiter=",")
    odom_train = np.loadtxt(open("odom_train.csv"), delimiter=",")
    odom_dt = np.loadtxt(open("odom_dt.csv"), delimiter=",")

    diff_dmag, diff_head, dmag_gt, dmag_x, dmag_y, dmag_h, odom_dt = viz_data(
        gt_dead, gt_train, odom_train)

    # plot_viz(odom_train, diff_dmag, diff_head, dmag_gt, dmag_x, dmag_y, dmag_h, odom_dt)

    xm, xmdt, ymabs, ymcart = setup(dmag_gt, dmag_h, dmag_x, dmag_y, odom_train, odom_dt)

    k = 0.03
    # perform LWLR
    yhat = lwlr(xmdt, xmdt, ymabs, k)

    yhat1, yhat2 = np.hsplit(yhat, 2)
    ymabs1, ymabs2 = np.hsplit(ymabs, 2)
    v, w, xm3 = np.hsplit(xmdt, 3)
    # test2, test3, test4 = np.hsplit(test1, 3)

    # print(np.shape(ymabs1))
    # print(np.shape(odom_train))

    plot(v, ymabs1, v, yhat1)


if __name__ == "__main__":
    main()