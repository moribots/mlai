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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv


class DataFrame():
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.timestamp = 0
        self.start = 0
        self.count = 0
        self.o = 0

    # Used in heap queue (priority queue for min)
    # def __lt__(self, neighbour):  # overload operator for heap queue
    #     # eval heapq based on this value
    #     return self.timestamp < neighbour.timestamp


def parse_dat(odometry, ground_truth):
    """ This function parses the odometry and ground_truth datasets
        to result in new sets in which there is only one odometry
        command for each change in ground truth position
    """
    new_odom = []
    new_gt = []
    odom = DataFrame(odometry)
    for t in range(len(ground_truth) - 1):
        odom.count = 0
        for o in range(odom.start, len(odometry)):

            if odometry[o][0] >= ground_truth[t][0] and odometry[o][
                    0] < ground_truth[t + 1][0]:
                odom.count += 1
            else:
                odom.start = o
                break
        if odom.count == 1:
            new_odom.append(odometry[o - 1])
            new_gt.append(ground_truth[t])

    new_odom.pop(-1)  # remove the last row

    return new_odom, new_gt


def get_gtr(odometry, ground_truth):
    """ compute deadreckoning difference
    """
    fwd_gt = [ground_truth[0]]
    odom = DataFrame(odometry)
    for t in range(1, len(ground_truth)):
        for o in range(odom.start, len(odometry)):
            if odometry[o][0] >= ground_truth[
                    t - 1][0] and odometry[o][0] < ground_truth[t][0]:
                # timestep = ground_truth[t][0] - odometry[o][0]
                timestep = ground_truth[t][0] - ground_truth[t - 1][0]
                fwd = fwd_prop(ground_truth[t - 1], odometry[o], timestep)
                fwd_gt.append(fwd)
                odom.start = o + 1  # try o
                break
    return fwd_gt


def fwd_prop(ground_truth, odometry, dt):
    """ Forward propagate the motion model
            using dead reckoning

            1 of 2 assumptions:
            1- odom applied between t_odom and t_gt
            starting at 0 vel
            2- odom applied between t-1_gt and t_gt
            starting at 0 vel
        """
    x = ground_truth[1] + odometry[1] * np.cos(ground_truth[3]) * dt
    y = ground_truth[2] + odometry[1] * np.sin(ground_truth[3]) * dt
    theta = ground_truth[3] + odometry[2] * dt
    # return new ground truth at time stamp
    return [dt + odometry[0], x, y, theta]


def viz_data(fwd_gt, gt, odom):
    """ Plot error in data


        NOTE:

        input [v w]^T * dt
        output [dx dy dtheta] (all ground truth)

        also assuming no prior controls and that controls extend to next gt
    """
    # start by turning data into 2D array
    fwd_gt = np.array(fwd_gt)
    gt = np.array(gt)
    odom = np.array(odom)

    n = np.shape(odom)[0]

    for i in range(n - 1):
        odom[i, 1:] = odom[i, 1:] * (gt[i + 1, 0] - gt[i, 0])

    # # now calculate distance magnitude for each set
    # dmag_fgt = np.sqrt(np.square(fwd_gt[:,1]) + np.square(fwd_gt[:,2]))
    dmag_x = gt[:, 1]
    dmag_y = gt[:, 2]
    dmag_h = gt[:, 3]
    dmag_gt = np.sqrt(np.square(dmag_x) + np.square(dmag_y))

    # now calculate x difference
    x_diff = fwd_gt[:, 1] - dmag_x
    y_diff = fwd_gt[:, 2] - dmag_y

    # now calculate distance magnitude difference
    diff_dmag = np.sqrt(np.square(x_diff) + np.square(y_diff))

    # now calculate heading difference for each set
    diff_head = fwd_gt[:, 3] - dmag_h

    return diff_dmag, diff_head, dmag_gt, dmag_x, dmag_y, dmag_h, odom


def plot(odom, diff_dmag, diff_head, dmag_gt, dmag_x, dmag_y, dmag_h, odom_dt):
    """ Plot error in delta_gt vs delta_gt_dreck
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

    # Initialize Plot dmagv
    plt.figure(1)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance error for v commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('v [m/s]')
    plt.scatter(odom_dt[:, 1], diff_dmag)

    # Initialize Plot dmagw
    plt.figure(2)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Distance error for w commands')
    plt.ylabel('dmag [m]')
    plt.xlabel('w [rad/s]')
    plt.scatter(odom_dt[:, 2], diff_dmag)

    # # Initialize Plot hmagv
    # plt.figure(3)
    # plt.autoscale(enable=True, axis='both', tight=None)
    # plt.title('Heading error for v commands')
    # plt.ylabel('hdiff [rad]')
    # plt.xlabel('v [m/s]')
    # plt.scatter(odom[:, 1], diff_head)

    # # Initialize Plot hmagw
    # plt.figure(4)
    # plt.autoscale(enable=True, axis='both', tight=None)
    # plt.title('Heading error for w commands')
    # plt.ylabel('hdiff [rad]')
    # plt.xlabel('w [rad/s]')
    # plt.scatter(odom[:, 2], diff_head)

    # # Initialize Plot dmag_gt
    # plt.figure(5)
    # plt.autoscale(enable=True, axis='both', tight=None)
    # plt.title('fig')
    # plt.ylabel('[m]')
    # plt.xlabel('v [m/s]')
    # plt.scatter(odom[:, 1], dmag_gt)

    # Initialize Plot dmag_x
    plt.figure(6)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('dmagx')
    plt.ylabel('[m]')
    plt.xlabel('v [m/s]')
    plt.scatter(odom_dt[:, 1], dmag_x)

    # Initialize Plot dmag_x
    plt.figure(7)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('dmagx')
    plt.ylabel('[m]')
    plt.xlabel('w [rad/s]')
    plt.scatter(odom_dt[:, 2], dmag_x)

    # Initialize Plot dmag_y
    plt.figure(8)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('dmagy')
    plt.ylabel('[m]')
    plt.xlabel('v [m/s]')
    plt.scatter(odom_dt[:, 1], dmag_y)

    # Initialize Plot dmag_y
    plt.figure(9)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('dmagy')
    plt.ylabel('[m]')
    plt.xlabel('w [rad/s]')
    plt.scatter(odom_dt[:, 2], dmag_y)

    # Initialize Plot dmag_h
    plt.figure(10)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('dmagh')
    plt.ylabel('[rad]')
    plt.xlabel('v [m/s]')
    plt.scatter(odom_dt[:, 1], dmag_h)

    # Initialize Plot dmag_h
    plt.figure(11)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('dmagh')
    plt.ylabel('[rad]')
    plt.xlabel('w [rad/s]')
    plt.scatter(odom_dt[:, 2], dmag_h)

    plt.show()


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


def main():
    odometry = read_dat(
        3, "ds0/ds0_Odometry.dat",
        ["Time [s]", "forward velocity [m/s]", "angular velocity[rad/s]"])
    ground_truth = read_dat(
        3, "ds0/ds0_Groundtruth.dat",
        ["Time [s]", "x [m]", "y [m]", "orientation [rad]"])

    # parse odometry and ground_truth data
    odom, gt = parse_dat(odometry, ground_truth)

    # calculate dead reckoning set
    fwd_gt = get_gtr(odom, gt)

    # compute error in dead reckoning set
    diff_dmag, diff_head, dmag_gt, dmag_x, dmag_y, dmag_h, odom_dt = viz_data(
        fwd_gt, gt, odom)

    # Create CSV File
    with open("odom_train.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(odom)
    with open("gt_train.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(gt)
    with open("gt_deadreck.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(fwd_gt)
    with open("odom_dt.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(odom_dt)

    # plot errors
    plot(odom, diff_dmag, diff_head, dmag_gt, dmag_x, dmag_y, dmag_h, odom_dt)


if __name__ == "__main__":
    main()