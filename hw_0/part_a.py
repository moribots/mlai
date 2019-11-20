#!/usr/bin/env python
# use chmod +x file_name.py (+x grants executable permission)
# use ./file_name.py to run
"""
Python 2.7 PEP8 Style
Code submission for Homework 0 Part A
Machine Learning & Artificial Intelligence for Robotics
Data Set 0
Particle Filter (used in Part B)

Data interpreted with Python's Pandas library

Maurice Rahme
Student ID: 3219435
mauricerahme2020@u.northwestern.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv


# Robot class contains position attributes (x, y, theta) and move method
class Robot_a():
    def __init__(self, position):
        self.position = position  # x, y, theta at world origin

    def move(self, control, t_next, noise_matrix):
        # Using abs for compatibility with a2 and a3
        """
        Note that noise is added to all motion: x,y,thetha
        So it is assumed that during pure rotation, the robot may
        slide, and its x,y position may change as a result
        """
        dt = np.abs(control[0] - t_next)
        x = self.position[0] + control[1] * np.cos(
            self.position[2]) * dt + noise_matrix[0][0]
        y = self.position[1] + control[1] * np.sin(
            self.position[2]) * dt + noise_matrix[1][1]
        theta = self.position[2] + control[2] * dt + noise_matrix[2][2]
        self.position = [x, y, theta]

    def measure_a6(self, position, landmark, noise_option):
        # Use standard deviation in ds0_Landmark_Groundtruth.dat
        if noise_option == 'y':
            mu = 0
            """
            The standard deviation of the Viscon measurement
            for the landmarks (ds0_Landmark_Groundtruth.dat)
            for each landmark is used to build the noise matrix
            of the measurement model
            """
            sigma_x = landmark[3]
            sigma_y = landmark[4]
            cov_x = (np.random.normal(mu, sigma_x))**2
            cov_y = (np.random.normal(mu, sigma_y))**2
            print('The added noise in x and y is {} and {}'.format(
                cov_x, cov_y))
        else:
            cov_x = 0
            cov_y = 0

        # Measure Range and Bearing
        # Note that cov_x and cov_y are added to the landmark cartersian
        # coordinates since this is the origin of the measured noise
        r2l_range = np.sqrt((position[0] - landmark[1] - cov_x)**2 +
                            (position[1] - landmark[2] - cov_y)**2)
        # arctan2 has built-in logic to account for quadrants
        r2l_bearing = np.arctan2((landmark[2] + cov_y - position[1]),
                                 (landmark[1] + cov_x - position[0]))
        # Measure x,y of landmark based on Range and Bearing
        x_l = position[0] + np.cos(r2l_bearing) * r2l_range
        y_l = position[1] + np.sin(r2l_bearing) * r2l_range
        rb = [r2l_range, r2l_bearing]
        xy = [x_l, y_l]
        rbxy = [rb, xy]
        return rbxy


# Exercise 2, plot robot movement using given commands
def a2(noise_option):
    # Initialize Robot instance
    position = [0, 0, 0]
    robot = Robot_a(position)
    path = []
    # Manual transcription of controls for Exercise 2
    control = [
        position, [1, 0.5, 0], [1, 0, -1 / (2 * np.pi)], [1, 0.5, 0],
        [1, 0, 1 / (2 * np.pi)], [1, 0.5, 0]
    ]
    # Initialize Noise Model
    if noise_option == 'y':
        # noise
        mu = 0
        # Note Viscon std_dev is 0.003, so assume worse than this
        sigma_x = 0.005
        x_noise = (np.random.normal(mu, sigma_x))**2  # square std for cov
        sigma_y = 0.005
        y_noise = (np.random.normal(mu, sigma_y))**2
        sigma_theta = 0.24
        theta_noise = (np.random.normal(mu, sigma_theta))**2
        noise_matrix = [[x_noise, 0, 0], [0, y_noise, 0], [0, 0, theta_noise]]
        print('The added noise in x,y,theta is: {}'.format(noise_matrix))
    else:
        # no noise
        noise_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # Initialize Plot
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Motion Iteration: 0')
    plt.ylabel('y [m]')
    plt.xlabel('x [m]')

    for t in range(len(control)):
        # Method in Robot class which updates pose based on controls
        # No dt_prev here since dt is defined
        # Noise matrix is optional based on user input
        robot.move(control[t], 0, noise_matrix)
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
def a3(odometry, ground_truth, range_response, noise_option):
    position = [1.29812900, 1.88315210,
                2.82870000]  # Set equal to Ground Truth Initial
    # Initialize Noise Model
    if noise_option == 'y':
        # noise
        mu = 0
        # Note Viscon std_dev is 0.003, so assume worse than this
        sigma_x = 0.005
        x_noise = (np.random.normal(mu, sigma_x))**2  # square std for cov
        sigma_y = 0.005
        y_noise = (np.random.normal(mu, sigma_y))**2
        sigma_theta = 0.012
        theta_noise = (np.random.normal(mu, sigma_theta))**2
        noise_matrix = [[x_noise, 0, 0], [0, y_noise, 0], [0, 0, theta_noise]]
        print('The added noise in x,y,theta is: {}'.format(noise_matrix))
    else:
        # no noise
        noise_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # Initialize Robot Instance
    robot = Robot_a(position)
    path = [robot.position]
    control = odometry

    for t in range(len(control) - 1):
        t_next = control[t + 1][0]

        robot.move(control[t], t_next, noise_matrix)
        path.append(robot.position)

    # Save path to .csv
    with open("dead_reckoning.csv", "w+") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(path)

    # Initialize Plot
    plt.autoscale(enable=True, axis='both', tight=None)
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
    if range_response == 'y':

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


# Exercise 6, test measurement model
def a6(landmark_groundtruth, noise_option):
    # Initialize Robot Instance
    position = [0, 0, 0]
    robot = Robot_a(position)
    # positions and landmark #s given in exercise
    positions = [[2, 3, 0], [0, 3, 0], [1, -2, 0]]
    landmarks = [6, 13, 17]
    landmark_list = []

    for l in range(len(landmarks)):
        for lg in range(len(landmark_groundtruth)):
            if landmark_groundtruth[lg][0] == landmarks[l]:
                landmark_list.append(landmark_groundtruth[lg])

    rb_list = []
    xy_list = []
    for p in range(len(positions)):
        rbxy = robot.measure_a6(positions[p], landmark_list[p], noise_option)
        rb = rbxy[0]
        xy = rbxy[1]
        rb_list.append(rb)
        xy_list.append(xy)

    err_list = []
    for i in range(len(xy_list)):
        x_err = abs(xy_list[i][0] - landmark_list[i][1])
        y_err = abs(xy_list[i][1] - landmark_list[i][2])
        xy_err = [x_err, y_err]
        err_list.append(xy_err)
    for p in range(len(err_list)):
        print('Measurement {}:'.format(p + 1))
        print('Error in x[m] and y[m]: {}'.format(err_list[p]))
        print('Range[m]: {}'.format(rb_list[p][0]))
        print('Bearing[m]: {}\n'.format(rb_list[p][1]))
    if noise_option != 'y':
        print(
            'Note that a nonzero, but infinitesmal (E-16) value may' +
            ' be returned due to the rounding of numpy.trigonometry functions')


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
    landmark_groundtruth = read_dat(
        3, "ds0/ds0_Landmark_Groundtruth.dat",
        ["Subject #", "x [m]", "y [m]", "x std-dev [m]", "y std-dev [m]"])
    # Select Exercise
    exercise = raw_input('Select an exercise [2,3,6]')
    if exercise == '2':
        # Exercise 2
        noise_option = raw_input('Add noise to the motion model? [y/n]')
        a2(noise_option)
    elif exercise == '3':
        # Exercise 3
        noise_option = raw_input('Add noise to the motion model? [y/n]')
        range_response = raw_input(
            'Show the first 2000 steps (start of major divergence)? [y/n]')
        a3(odometry, ground_truth, range_response, noise_option)
    elif exercise == '6':
        # Exercise 6
        noise_option = raw_input('Add noise to the measurement model? [y/n]')
        a6(landmark_groundtruth, noise_option)
    else:
        print('You did not select an exercise, please try again.')
    # Calculating average dt
    diff_list = []
    for l in range(len(odometry) - 1):
        diff = odometry[l + 1][0] - odometry[l][0]
        diff_list.append(diff)
    average_dt = sum(diff_list) / len(diff_list)
    average_hz = 1 / average_dt
    print(average_dt)
    print(average_hz)

    plt.show()


if __name__ == "__main__":
    main()
