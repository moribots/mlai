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


def lwlr_pt(x_q, xm, ym, k, xval, dim):
    # convert to matrix
    # xM = np.mat(xm)
    # yM = np.mat(ym)
    xM = np.reshape(xm, (-1, 3))
    yM = np.reshape(ym, (-1, dim))
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
    r = np.empty((len(xM), dim))
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
    dim = np.shape(ym)[1]
    m = np.shape(test)[0]
    y_hat = np.zeros((m, dim))
    MSE = np.zeros((m, dim))
    VAR = np.zeros((m, dim))
    for i in range(m):
        # find Beta and hence y_hat for every x_q (test[i])
        y_hat[i], MSE[i], VAR[i] = lwlr_pt(test[i], xm, ym, k, False, dim)
        if i % 200 == 0:
            print("Completed {} of {}".format(i, m))
    return y_hat, MSE, VAR


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
    num = 100  # 5000
    xm = xm[:num, :]
    xmdt = xmdt[:num, :]
    ymabs = ymabs[:num, :]
    ymcart = ymcart[:num, :]

    # Use first few samples from test data
    xmtest = xmtest[:3000, :]

    return xm, xmdt, ymabs, ymcart, xmtest


def train_test_split(dataset, split=0.80):
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

    for d in range(len(gt_dead) - 1):
        gt_dead[d, :] = gt_dead[d + 1, :] - gt_dead[d, :]

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
    # x = path[0] + yhat[0]
    # y = path[1] + yhat[1]
    # theta = path[2] + yhat[2]
    return x, y, theta


def pos_err_var(gt_train, gt_dead, odom_train, odom_test, ground_truth):
    diff_dmag, diff_head, dmag_gt, dmag_x, dmag_y, dmag_h, odom_dt, odom_test = viz_data(
        gt_dead, gt_train, odom_train, odom_test, ground_truth)

    # First turn odom into array
    odom_dt = np.array(odom_dt)

    # remove first datapoint from diff_dmag and diff_head
    diff_dmag = np.delete(diff_dmag, (0), axis=0)
    diff_head = np.delete(diff_head, (0), axis=0)
    dmag_gt = np.delete(dmag_gt, (0), axis=0)
    dmag_x = np.delete(dmag_x, (0), axis=0)
    dmag_y = np.delete(dmag_y, (0), axis=0)
    dmag_h = np.delete(dmag_h, (0), axis=0)

    # Variance
    print("The variance on the position error is {}".format(np.var(diff_dmag)))
    print("The variance on the heading error is {}".format(np.var(diff_head)))

    plt.figure(1)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance error for v commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('vdt [m]')
    plt.scatter(odom_dt[:, 1], diff_dmag)

    plt.figure(2)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance error for w commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('wdt [rad]')
    plt.scatter(odom_dt[:, 2], diff_dmag)

    plt.figure(3)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Heading error for v commands')
    plt.ylabel('dhead [rad]')
    plt.xlabel('vdt [m]')
    plt.scatter(odom_dt[:, 1], diff_head)

    plt.figure(4)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Heading error for w commands')
    plt.ylabel('dhead [rad]')
    plt.xlabel('wdt [rad]')
    plt.scatter(odom_dt[:, 2], diff_head)

    plt.figure(5)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance change for v commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('vdt [m]')
    plt.scatter(odom_dt[:, 1], dmag_gt)

    plt.figure(6)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance change for w commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('wdt [rad]')
    plt.scatter(odom_dt[:, 2], dmag_gt)

    plt.figure(7)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Heading change for v commands')
    plt.ylabel('dhead [rad]')
    plt.xlabel('vdt [m]')
    plt.scatter(odom_dt[:, 1], dmag_h)

    plt.figure(8)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Heading change for w commands')
    plt.ylabel('dhead [rad]')
    plt.xlabel('wdt [rad]')
    plt.scatter(odom_dt[:, 2], dmag_h)

    plt.show()


def remove_outliers(gt_train, gt_dead, odom_train, odom_test, ground_truth):
    diff_dmag, diff_head, dmag_gt, dmag_x, dmag_y, dmag_h, odom_dt, odom_test = viz_data(
        gt_dead, gt_train, odom_train, odom_test, ground_truth)

    # First turn odom into array
    odom_dt = np.array(odom_dt)
    # Max dmag_gt is 0.013
    # Max dmag_h is 0.10
    ax = 1
    gh = 0
    while gh < len(dmag_gt) - ax:
        if abs(dmag_gt[gh]) > 0.013 or abs(dmag_h[gh]) > 1:
            dmag_gt = np.delete(dmag_gt, gh, axis=0)
            odom_dt = np.delete(odom_dt, gh, axis=0)
            dmag_h = np.delete(dmag_h, gh, axis=0)
            dmag_x = np.delete(dmag_x, gh, axis=0)
            dmag_y = np.delete(dmag_y, gh, axis=0)
            # print("deleted")
            ax += 1
            gh -= 1
        gh += 1
        # if gh % 1000 == 0:
        #     print(gh)
    # Now do the same for the vertical spikes at 0 for w
    ax = 1
    gh = 0
    while gh < len(dmag_gt) - ax:
        if abs(dmag_gt[gh]) > 0.003 or abs(dmag_h[gh]) > 0.003:
            if odom_dt[gh, 2] > -0.001 and odom_dt[gh, 2] < 0.001 or odom_dt[
                    gh, 1] < 0.00001:
                dmag_gt = np.delete(dmag_gt, gh, axis=0)
                odom_dt = np.delete(odom_dt, gh, axis=0)
                dmag_h = np.delete(dmag_h, gh, axis=0)
                dmag_x = np.delete(dmag_x, gh, axis=0)
                dmag_y = np.delete(dmag_y, gh, axis=0)
                # print("deleted")
                ax += 1
                gh -= 1
        gh += 1

    # Now print new lengths for check
    print(np.shape(dmag_gt))
    print(np.shape(dmag_h))

    lim = 5000

    plt.figure(5)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance change for v commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('vdt [m]')
    plt.scatter(odom_dt[:lim, 1], dmag_gt[:lim])

    plt.figure(6)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance change for w commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('wdt [rad]')
    plt.scatter(odom_dt[:lim, 2], dmag_gt[:lim])

    plt.figure(7)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Heading change for v commands')
    plt.ylabel('dhead [rad]')
    plt.xlabel('vdt [m]')
    plt.scatter(odom_dt[:lim, 1], dmag_h[:lim])

    plt.figure(8)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Heading change for w commands')
    plt.ylabel('dhead [rad]')
    plt.xlabel('wdt [rad]')
    plt.scatter(odom_dt[:lim, 2], dmag_h[:lim])

    # plt.show()

    return diff_dmag, diff_head, dmag_gt, dmag_x, dmag_y, dmag_h, odom_dt, odom_test


def lwlr_crossval(test, xm, ym, k):
    """ xm: nx(m+1) (col of 1s at end)
        ym: nxm or nx1 for singular
        Beta: (n+1)xm
        test: (n+1)xm (row of 1 at end)
    """
    dim = np.shape(ym)[1]
    m = np.shape(test)[0]
    y_hat = np.zeros((m, dim))
    MSE = np.zeros((m, dim))
    VAR = np.zeros((m, dim))
    for i in range(m):
        # find Beta and hence y_hat for every x_q (test[i])
        y_hat[i], MSE[i], VAR[i] = lwlr_pt(test[i], xm, ym, k, True, dim)
        if i % 200 == 0:
            print("Completed {} of {}".format(i, m))
    return y_hat, MSE, VAR


def lwlr_main(gt_train, gt_dead, odom_train, odom_dt, odom_test, ground_truth, LWLR_m):

    diff_dmag, diff_head, dmag_gt, dmag_x, dmag_y, dmag_h, odom_dt, odom_test = remove_outliers(
        gt_train, gt_dead, odom_train, odom_test, ground_truth)

    # plot_viz(odom_train, diff_dmag, diff_head, dmag_gt, dmag_x, dmag_y, dmag_h, odom_dt)

    xm, xmdt, ymabs, ymcart, xmtest = setup(dmag_gt, dmag_h, dmag_x, dmag_y,
                                            odom_train, odom_dt, odom_test)

    k = 0.0005  # 8e-05

    if LWLR_m == 0:
        # perform LWLR
        yhat, MSE, VAR = lwlr(xmtest, xmdt, ymabs, k)
        print(np.shape(MSE))
        print(np.shape(VAR))

        # with open("yhat.csv", "w+") as my_csv:
        #     csvWriter = csv.writer(my_csv, delimiter=',')
        #     csvWriter.writerows(yhat)

        # with open("xmtest.csv", "w+") as my_csv:
        #     csvWriter = csv.writer(my_csv, delimiter=',')
        #     csvWriter.writerows(xmtest)

        # with open("xmdt.csv", "w+") as my_csv:
        #     csvWriter = csv.writer(my_csv, delimiter=',')
        #     csvWriter.writerows(xmdt)

        path = [[1.29812900, 1.88315210, 2.82870000]]

        for i in range(len(xmtest)):
            x, y, theta = move(yhat[i, :], path[-1])
            path.append([x, y, theta])

        # Plot lwlr vs gt
        # Initialize Plot
        plt.figure(100)
        plt.autoscale(enable=True, axis='both', tight=None)
        plt.title('LWLR Pose Estimation VS. Ground Truth Data')
        plt.ylabel('y [m]')
        plt.xlabel('x [m]')

        # Set range for desired final iteration
        inc_range = 3000
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
        test2, test3, test4 = np.hsplit(test1, 3)

        # print(np.shape(ymabs1))
        # print(np.shape(odom_train))

        # plot(v, ymabs1, v, yhat1)

    if LWLR_m == 1:
        # perform xvalidation
        yhat_x, MSE_x, VAR_x = lwlr_crossval(xmdt, xmdt, ymabs, k)

        MSE_xd, MSE_xh = np.hsplit(MSE_x, 2)
        VAR_xd, VAR_xh = np.hsplit(VAR_x, 2)

        MSE_d = np.sum(MSE_xd)
        MSE_h = np.sum(MSE_xh)

        VAR_d = np.sum(VAR_xd)
        VAR_h = np.sum(VAR_xh)

        MSE_xd = MSE_xd.flatten()
        MSE_xh = MSE_xh.flatten()

        VAR_xd = VAR_xd.flatten()
        VAR_xh = VAR_xd.flatten()

        MSE_xd = MSE_xd.tolist()
        MSE_xh = MSE_xh.tolist()

        print(len(MSE_xd))

        # VAR_xd = VAR_xd.tolist()
        # VAR_xh = VAR_xd.tolist()

        print("h is {}".format(k))
        print("MSE sum for dmag: {}".format(MSE_d))
        print("MSE sum for hmag: {}".format(MSE_h))

        # Plot MSE
        plt.figure(101)
        plt.title("Mean Square Error for each x_q")
        plt.ylabel('MSE')
        plt.xlabel('x_q element')
        plt.plot(range(len(MSE_xd)),
                 MSE_xd,
                 color='b',
                 label='MSE for distance')
        plt.plot(range(len(MSE_xh)),
                 MSE_xh,
                 color='r',
                 label='MSE for heading')

        # Plot VAR
        plt.figure(102)
        plt.title("Variance for each x_q")
        plt.ylabel('VAR')
        plt.xlabel('x_q element')
        plt.plot(range(len(VAR_xd)),
                 VAR_xd,
                 color='b',
                 label='VAR for distance')
        plt.plot(range(len(VAR_xh)),
                 VAR_xh,
                 color='r',
                 label='VAR for heading')
        plt.legend()
        plt.show()

        print("VAR sum for dmag: {}".format(VAR_d))
        print("VAR sum for hmag: {}".format(VAR_h))
        # print(np.shape(yhat_x))
        # print(np.shape(MSE_x))
        # print(np.shape(VAR_x))


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

    # Plot positional error and variance
    # pos_err_var(gt_train, gt_dead, odom_train, odom_test, ground_truth)

    # Remove outliers
    # diff_dmag, diff_head, dmag_gt, dmag_x, dmag_y, dmag_h, odom_dt, odom_test = remove_outliers(
    #     gt_train, gt_dead, odom_train, odom_test, ground_truth)

    # LWLR
    # LWLR Mode: 0, normal, 1, xval
    LWLR_m = 1
    lwlr_main(gt_train, gt_dead, odom_train, odom_dt, odom_test, ground_truth, LWLR_m)


if __name__ == "__main__":
    main()