#!/usr/bin/env python
"""
Python 2.7 PEP8 Style
Code submission for Homework 1 Part A
Machine Learning & Artificial Intelligence for Robotics
Data Set 1
A* Search Implementation

Data interpreted with Python's Pandas library

Maurice Rahme
mauricerahme2020@u.northwestern.edu
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib.patches as patches
from matplotlib.path import Path
import pandas as pd


# Robot class for part a
class Robot():
    def __init__(self):
        self.position = 0


# Grid class
class Grid():
    """DOCSTRING
    """
    def __init__(self, cell_size, landmarks):
        self.xmin, self.xmax = -2, 6
        self.ymin, self.ymax = -6, 6
        self.cell_size = cell_size
        self.cell_xmin = np.arange(self.xmin, self.xmax, self.cell_size)
        self.cell_ymin = np.arange(self.ymin, self.ymax, self.cell_size)
        self.centres = np.ones((len(self.cell_xmin), len(self.cell_ymin)))
        self.landmarks = landmarks
        self.prev_x = 0
        self.prev_y = 0
        self.obstacles()

    def obstacles(self):
        """DOCSTRING
        """
        for ob in range(len(self.landmarks)):
            index_x = []
            index_y = []
            for x in range(len(self.cell_xmin)):
                # print("x is {}".format(x))
                if self.landmarks[ob][0] >= self.cell_xmin[
                        x] and self.landmarks[ob][0] <= self.cell_xmin[x] + 1:
                    index_x.append(x)

            for y in range(len(self.cell_ymin)):
                if self.landmarks[ob][1] >= self.cell_ymin[
                        y] and self.landmarks[ob][1] <= self.cell_ymin[y] + 1:
                    index_y.append(y)

            if len(index_x) == 1 and len(index_y) == 1:
                self.centres[index_x, index_y] = 1000  # cost of obstacle
            elif len(index_x) > len(index_y):
                for indx in range(len(index_x)):
                    for indy in range(len(index_y)):
                        self.centres[index_x[indx], index_y[indy]] = 1000
            elif len(index_y) > len(index_x):
                for indx in range(len(index_y)):
                    for indy in range(len(index_x)):
                        self.centres[index_x[indx], index_y[indy]] = 1000


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


def plot(landmark_list):
    # Initialize grid with size and landmarks
    a_grid = Grid(1, landmark_list)

    # Initialise Plot
    fig, ax = plt.subplots()

    # plt.imshow adds colour to higher values
    # used to show occupied cells
    plt.imshow(a_grid.centres.T,
               cmap="Purples",
               origin='lower',
               extent=[a_grid.xmin, a_grid.xmax, a_grid.ymin, a_grid.ymax])
    # extent is left, right, bottom, top

    # Set axis ranges
    ax.set_xlim(a_grid.xmin, a_grid.xmax)
    ax.set_ylim(a_grid.ymin, a_grid.ymax)

    # Change major ticks
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Change minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator(0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(0.1))

    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='darkviolet', linestyle='--')
    ax.grid(which='minor', color='darkviolet', linestyle='--')

    plt.axis([-2, 5, -6, 6])
    plt.xlabel('x position [m]')
    plt.ylabel('y position [m]')

    plt.show()


# Main
def main():
    # Load Data from ds1 set using Pandas
    landmark_groundtruth = read_dat(
        3, "ds1/ds1_Landmark_Groundtruth.dat",
        ["Subject #", "x [m]", "y [m]", "x std-dev [m]", "y std-dev [m]"])
    landmark_list = []
    for l in range(len(landmark_groundtruth)):
        landmark_list.append(
            [landmark_groundtruth[l][1], landmark_groundtruth[l][2]])

    plot(landmark_list)
    """

    # Select Exercise
    exercise = raw_input('Select an exercise [1,3,5,6,7]')
    if exercise == '1':
        # Exercise 1
        x = 1
    """


if __name__ == "__main__":
    main()
