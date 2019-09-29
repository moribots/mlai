#!/usr/bin/env python
# use chmod +x file_name.py (+x grants executable permission)
# use ./file_name.py to run
"""
Python 2.7 PEP8 Style
Code submission for Homework 0 Part A
Machine Learning & Artificial Intelligence for Robotics
Data Set 0
Particle Filter

Maurice Rahme
Student ID: 3219435
mauricerahme2020@u.northwestern.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Robot class contains position attributes (x, y, theta) and move method
class Robot():
    def __init__(self, position):
        self.position = position  # x, y, theta at world origin

    def move(self, control, t_next):
        # Using abs for compatibility with a2 and a3
        dt = np.abs(control[0] - t_next)
        x = self.position[0] + control[1] * np.cos(self.position[2]) * dt
        y = self.position[1] + control[1] * np.sin(self.position[2]) * dt
        theta = self.position[2] + control[2] * dt
        self.position = [x, y, theta]


# Exercise 2, plot robot movement using given commands
def a2():
    # Initialize Robot instance
    position = [0, 0, 0]
    robot = Robot(position)
    path = []
    # Manual transcription of controls for Exercise 2
    control = [
        position, [1, 0.5, 0], [1, 0, -1 / (2 * np.pi)], [1, 0.5, 0],
        [1, 0, 1 / (2 * np.pi)], [1, 0.5, 0]
    ]

    # Initialize Plot
    plt.xlim((-0.2, 2))
    plt.ylim((-0.5, 0.2))
    plt.title('Motion Iteration: 0')
    plt.ylabel('y [m]')
    plt.xlabel('x [m]')

    for t in range(len(control)):
        # Method in Robot class which updates pose based on controls
        robot.move(control[t], 0)  # No dt_prev here since dt is defined
        path.append(robot.position)
        plt.title('Motion Iteration: ' + str(t - 1))
        if t == 0:
            plt.plot(robot.position[0], robot.position[1], 'ko', markersize=10)
        else:
            plt.plot([path[t - 1][0], path[t][0]],
                     [path[t - 1][1], path[t][1]], '-kd')
        plt.pause(0.05)

    # Plot final position clearly
    plt.plot(robot.position[0],
             robot.position[1],
             color='darkviolet',
             marker='o',
             markersize=10)
    return robot


# Exercise 3, plot robot movement using Odometry.dat
def a3(odometry, ground_truth, response):
    position = [1.29812900, 1.88315210,
                2.82870000]  # Set equal to Ground Truth Initial
    robot = Robot(position)
    path = [robot.position]
    control = odometry

    for t in range(len(control) - 1):
        t_next = control[t + 1][0]

        robot.move(control[t], t_next)
        path.append(robot.position)

    # Initialize Plot
    if response == 'y':
        xmin = 0
        xmax = 2
        ymin = 1
        ymax = 2.8
    else:
        xmin = -1
        xmax = 12
        ymin = -3.5
        ymax = 3.5
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.title('Dead Reckoning Pose Estimation VS. Ground Truth Data')
    plt.ylabel('y [m]')
    plt.xlabel('x [m]')

    # Set range for desired final iteration
    inc_range = 2000
    # Plot Dead Reckoning
    path_x = [x[0] for x in path]
    path_y = [y[1] for y in path]
    # Plot Ground Truth Data
    ground_truth_x = [x[1] for x in ground_truth]
    ground_truth_y = [y[2] for y in ground_truth]
    # Optional Range Reduction
    if response == 'y':

        # Dead Reckoning
        path_xs = []
        path_ys = []
        for px in range(inc_range):
            path_xs.append(path_x[px])
        for py in range(inc_range):
            path_ys.append(path_y[py])
        plt.plot(path_xs, path_ys, '-k', label='Dead Reckoning Data')
        # Append final index of reduced range to
        # full range for plotting
        path_x.append(path_xs[-1])
        path_y.append(path_ys[-1])

        # Ground Truth
        ground_truth_xs = []
        ground_truth_ys = []
        for gx in range(inc_range):
            ground_truth_xs.append(ground_truth_x[gx])
        for gy in range(inc_range):
            ground_truth_ys.append(ground_truth_y[gy])
        plt.plot(ground_truth_xs,
                 ground_truth_ys,
                 '-g',
                 label='Ground Truth Data')
        # Append final index of reduced range to
        # full range for plotting
        ground_truth_x.append(ground_truth_xs[-1])
        ground_truth_y.append(ground_truth_ys[-1])
    else:
        plt.plot(path_x, path_y, '-k', label='Dead Reckoning Data')
        plt.plot(ground_truth_x,
                 ground_truth_y,
                 '-g',
                 label='Ground Truth Data')

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

    return robot


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


# Main
def main():
    # Load Data from ds0 set using Pandas
    odometry = read_dat(
        3, "ds0/ds0_Odometry.dat",
        ["Time [s]", "forward velocity [m/s]", "angular velocity[rad/s]"])
    ground_truth = read_dat(
        3, "ds0/ds0_Groundtruth.dat",
        ["Time [s]", "x [m]", "y [m]", "orientation [rad]"])
    # Select Exercise
    try:
        exercise = raw_input('Select an exercise [2,3,6]')
        if exercise != '3' or '3' or '6':
            raise ValueError(
                'You did not select an available exercise. Please try again.')
    except ValueError:
        print('Please select exercise 2, 3, or 6.')
    if exercise == '2':
        # Exercise 2
        a2()
    elif exercise == '3':
        # Exercise 3
        response = raw_input(
            'Show the first 2000 steps (start of major divergence)? [y/n]')
        a3(odometry, ground_truth, response)

    plt.show()


if __name__ == "__main__":
    main()
