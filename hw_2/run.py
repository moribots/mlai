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
from random import seed, randrange
import csv
import pandas as pd
import matplotlib.pyplot as plt


def lwlr_pt(x_q, xm, ym, k):
    # convert to matrix
    # xM = np.mat(xm)
    # yM = np.mat(ym)
    xM = np.reshape(xm, (-1, 3))
    yM = np.reshape(ym, (-1, 2))
    # diagonal matrix
    m = np.shape(xM)[0]
    # w = np.mat(np.eye((m)))
    w = np.eye(m)
    # fill weights using Gaussian Kernel
    for i in range(m):
        # diffM = x_q - xM[i, :]
        diffM = xM[i, :] - x_q
        dh = np.divide(np.dot(diffM.T, diffM), k)
        kernel = np.exp(-np.dot(dh, dh.T))
        w[i, i] = np.sqrt(kernel)
        # print(diffM)
        # w[i, i] = np.exp(np.dot(diffM, diffM.T) / (-2.0 * k**2))
        # w[i, i] = np.sqrt(np.dot(diffM.T, diffM)) / k

    # Find Beta
    Z = np.dot(w, xM)
    v = np.dot(w, yM)
    ZTZ = np.dot(Z.T, Z)
    # Try for inverse case
    try:
        # inverse
        inv = np.linalg.inv(ZTZ)
    except:
        # pseudoinverse
        print('pinv')
        inv = np.linalg.pinv(ZTZ)

    B = np.dot(np.dot(inv, Z.T), v)
    return np.dot(x_q.T, B)


def lwlr(test, xm, ym, k):
    """ xm: nx(m+1) (col of 1s at end)
        ym: nxm or nx1 for singular
        Beta: (n+1)xm
        test: (n+1)xm (row of 1 at end)
    """
    m = np.shape(test)[0]
    y_hat = np.zeros((m, 2))
    for i in range(m):
        # find Beta and hence y_hat for every x_q (test[i])
        y_hat[i] = lwlr_pt(test[i], xm, ym, k)
        print("Completed {} of {}".format(i, m))
    return y_hat


def setup(dmag_gt, dmag_h, dmag_x, dmag_y, odom_train, odom_dt, odom_test):

    # First turn odom into array
    odom_train = np.array(odom_train)
    odom_dt = np.array(odom_dt)
    odom_test = np.array(odom_test)

    # Extract commands (independent vars)
    v = odom_train[:, 1]
    w = odom_train[:, 2]

    vdt = odom_dt[:, 1]
    wdt = odom_dt[:, 2]

    vtest = odom_test[:, 1]
    wtest = odom_test[:, 2]

    # Reshape
    v = np.reshape(v, (-1, 1))
    vdt = np.reshape(vdt, (-1, 1))
    vtest = np.reshape(vtest, (-1, 1))

    w = np.reshape(w, (-1, 1))
    wdt = np.reshape(wdt, (-1, 1))
    wtest = np.reshape(wtest, (-1, 1))

    dmag_gt = np.reshape(dmag_gt, (-1, 1))
    dmag_x = np.reshape(dmag_x, (-1, 1))
    dmag_y = np.reshape(dmag_y, (-1, 1))
    dmag_h = np.reshape(dmag_h, (-1, 1))

    # Now create xm (input matrix)
    xm = np.hstack((v, w))
    xmdt = np.hstack((vdt, wdt))
    xmtest = np.hstack((vtest, wtest))

    # Now create ym (output matrix)
    ymabs = np.hstack((dmag_gt, dmag_h))
    ymcart = np.hstack((dmag_x, dmag_y, dmag_h))

    # TODO:
    # Add test input here, for now use train

    # Add ones col at end of inputs
    ones_col = np.ones((np.shape(xm)[0], 1))
    ones_coldt = np.ones((np.shape(xmdt)[0], 1))
    ones_coltest = np.ones((np.shape(xmtest)[0], 1))
    xm = np.hstack((xm, ones_col))
    xmdt = np.hstack((xmdt, ones_coldt))
    xmtest = np.hstack((xmtest, ones_coltest))

    # Now limit to number of points (lest lwlr take too long)
    num = 500
    xm = xm[:num, :]
    xmdt = xmdt[:num, :]
    ymabs = ymabs[:num, :]
    ymcart = ymcart[:num, :]

    # Use first few samples from test data
    xmtest = xmtest[:6000, :]

    return xm, xmdt, ymabs, ymcart, xmtest


def train_test_split(dataset, split=0.60):
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


def viz_data(gt_dead, gt_train, odom_train, odom_test, ground_truth):
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
    odom_test = np.array(odom_test)

    n = np.shape(odom_train)[0]

    for i in range(n - 1):
        odom_dt[i, 1:] = odom_dt[i, 1:] * (gt_train[i + 1, 0] - gt_train[i, 0])

    for g in range(len(gt_train) - 1):
        gt_train[g, :] = gt_train[g + 1, :] - gt_train[g, :]

    for od in range(np.shape(odom_test)[0] - 1):
        odom_test[od, 1:] = odom_test[od, 1:] * (odom_test[od + 1, 0] -
                                                 odom_test[od, 0])

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

    return diff_dmag, diff_head, dmag_gt_train, dmag_x, dmag_y, dmag_h, odom_dt, odom_test


# Read .dat Files using Pandas
def read_dat(start_index, file_path, usecols):
    # Read Data using Pandas
    data_str = pd.read_table(file_path,
                             sep="\s+",
                             skiprows=1,
                             usecols=usecols,
                             names=usecols)
    # Format Data into List
    data_str = data_str.values.tolist()

    # Useful data starts on start_index, preceeded by headings
    # Turn string data into floats, 64bit accuracy
    data = []
    for i in range(start_index, len(data_str)):
        data.append(np.array(data_str[i], dtype=np.float64))

    return data


def move(yhat, path):
    theta = path[2] + yhat[1]
    x = path[0] + yhat[0] * np.cos(path[2])
    y = path[1] + yhat[0] * np.sin(path[2])

    return x, y, theta


def main():
    gt_train = np.loadtxt(open("gt_train.csv"), delimiter=",")
    gt_dead = np.loadtxt(open("gt_deadreck.csv"), delimiter=",")
    odom_train = np.loadtxt(open("odom_train.csv"), delimiter=",")
    odom_dt = np.loadtxt(open("odom_dt.csv"), delimiter=",")
    odom_test = read_dat(
        3, "ds0/ds0_Odometry.dat",
        ["Time [s]", "forward velocity [m/s]", "angular velocity[rad/s]"])
    ground_truth = read_dat(
        3, "ds0/ds0_Groundtruth.dat",
        ["Time [s]", "x [m]", "y [m]", "orientation [rad]"])

    diff_dmag, diff_head, dmag_gt, dmag_x, dmag_y, dmag_h, odom_dt, odom_test = viz_data(
        gt_dead, gt_train, odom_train, odom_test, ground_truth)

    # plot_viz(odom_train, diff_dmag, diff_head, dmag_gt, dmag_x, dmag_y, dmag_h, odom_dt)

    xm, xmdt, ymabs, ymcart, xmtest = setup(dmag_gt, dmag_h, dmag_x, dmag_y,
                                            odom_train, odom_dt, odom_test)

    # dataset1 = np.hstack((xm, ymabs))
    # dataset2 = np.hstack((xm, ymcart))

    # dataset3 = np.hstack((xmdt, ymabs))
    # dataset4 = np.hstack((xmdt, ymcart))

    # train, test = train_test_split(dataset1, 0.60)

    # print(np.shape(train))
    # print(np.shape(test))

    k = 0.000095
    # perform LWLR
    yhat = lwlr(xmtest, xmdt, ymabs, k)

    with open("yhat.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(yhat)

    with open("xmtest.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(xmtest)

    with open("xmdt.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(xmdt)

    path = [[1.29812900, 1.88315210, 2.82870000]]

    for i in range(len(xmtest)):
        x, y, theta = move(yhat[i, :], path[-1])
        path.append([x, y, theta])

    # Plot lwlr vs gt
    # Initialize Plot
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Dead Reckoning Pose Estimation VS. Ground Truth Data')
    plt.ylabel('y [m]')
    plt.xlabel('x [m]')

    # Set range for desired final iteration
    inc_range = 6000
    # Plot lwlr
    path_x = [px[0] for px in path]
    path_y = [py[1] for py in path]
    plt.plot(path_x, path_y, '-k', label='LWLR Data')
    # Plot Ground Truth Data
    ground_truth_x = [gx[1] for gx in ground_truth]
    ground_truth_y = [gy[2] for gy in ground_truth]

    # Ground Truth
    ground_truth_xs = []
    ground_truth_ys = []
    for ggx in range(inc_range):
        ground_truth_xs.append(ground_truth_x[ggx])
    for ggy in range(inc_range):
        ground_truth_ys.append(ground_truth_y[ggy])
    plt.plot(ground_truth_xs, ground_truth_ys, '-g', label='Ground Truth Data')
    # Append final index of reduced range to
    # full range for plotting
    ground_truth_x.append(ground_truth_xs[-1])
    ground_truth_y.append(ground_truth_ys[-1])

    # Plot inital position (Both)
    plt.plot(path_x[0],
             path_y[0],
             color='gold',
             marker='o',
             markersize=10,
             label='Starting Point')

    # Plot final position (Dead Reckoning)
    plt.plot(path_x[-1],
             path_y[-1],
             color='darkviolet',
             marker='o',
             markersize=5,
             label='Endpoints')

    # Plot final position (Ground Truth)
    plt.plot(ground_truth_x[-1],
             ground_truth_y[-1],
             color='darkviolet',
             marker='o',
             markersize=5)

    # Show Legend
    plt.legend()

    plt.show()

    yhat1, yhat2 = np.hsplit(yhat, 2)
    ymabs1, ymabs2 = np.hsplit(ymabs, 2)
    v, w, xm3 = np.hsplit(xmdt, 3)
    # test2, test3, test4 = np.hsplit(test1, 3)

    # print(np.shape(ymabs1))
    # print(np.shape(odom_train))

    # plot(v, ymabs1, v, yhat1)


if __name__ == "__main__":
    main()