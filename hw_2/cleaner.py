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
    def __init__(self, dataframe, index):
        self.dataframe = dataframe
        self.timestamp = self.dataframe[index][0]
        self.start = 0
        self.count = 0
        self.o = 0

    # Used in heap queue (priority queue for min)
    def __lt__(self, neighbour):  # overload operator for heap queue
        # eval heapq based on this value
        return self.timestamp < neighbour.timestamp


def parse_dat(odometry, ground_truth):
    new_odom = []
    new_gt = []
    odom = DataFrame(odometry, 0)
    for t in range(len(ground_truth) - 1):
        odom.count = 0
        for o in range(odom.start, len(odometry) - 1):
            if odometry[o][0] >= ground_truth[t][0] and odometry[o][
                    0] <= ground_truth[t + 1][0]:
                odom.count += 1
            else:
                odom.start = o
                break
        if odom.count == 1:
            new_odom.append(odometry[o])
            new_gt.append(ground_truth[t])
    return new_odom, new_gt


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
    odom, gt = parse_dat(odometry, ground_truth)

    # Create CSV File
    with open("odom.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(odom)
    with open("gt.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(gt)




if __name__ == "__main__":
    main()