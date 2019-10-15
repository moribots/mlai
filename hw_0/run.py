#!/usr/bin/env python
"""
Python 2.7 PEP8 Style
Code submission for Homework 0 Part B
Machine Learning & Artificial Intelligence for Robotics
Data Set 0
Particle Filter Implementation

Data interpreted with Python's Pandas library

Maurice Rahme
mauricerahme2020@u.northwestern.edu
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Robot class for part a
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


# Robot class for part b
class Robot():
    def __init__(self, position, num_particles, sensor_noise, motion_noise):
        self.position = position  # x, y, theta at world origin
        self.M = num_particles  # number of particles used in filter
        # empty numpy array w/N rows, 4 columns for x,y,theta, weight
        self.particles = np.zeros((self.M, 4))
        # set initial particle weights (4th col) to be all equal and sum to 1
        self.particles[:, 3] = 1.0 / self.M
        # Initialize sensor noise
        self.sensor_noise = [
            np.random.normal(0, sensor_noise[0]),
            np.random.normal(0, sensor_noise[1]),
            np.random.normal(0, sensor_noise[2])
        ]
        # Initialize motion noise
        self.motion_noise = [
            np.random.normal(0, motion_noise[0]),
            np.random.normal(0, motion_noise[1])
        ]
        # Record last measurement to avoid reading list from scratch
        self.last_measurement = 0

    def measure(self, landmark, i):
        """
        Returns expected range and bearing measurement for each particle
        position as reported in section 1.5.
        """
        mu = 0
        # Use the std in Measurement.dat to compute range and bearing
        sigma_x = landmark[3]
        sigma_y = landmark[4]
        cov_x = (np.random.normal(mu, sigma_x))**2
        cov_y = (np.random.normal(mu, sigma_y))**2
        r2l_range = np.sqrt(
            np.power(self.particles[i, 0] - landmark[1] - cov_x, 2) +
            np.power(self.particles[i, 1] - landmark[2] - cov_y, 2))
        # arctan2 has built-in logic to account for quadrants
        r2l_bearing = np.arctan2((landmark[2] + cov_y - self.particles[i, 1]),
                                 (landmark[1] + cov_x - self.particles[i, 0]))
        rb = [r2l_range, r2l_bearing]
        return rb

    def init_known_particles(self, std):
        """ USE IF INITIAL CONDITION (STATE) KNOWN

            Generates a gaussian distribution of particles about the
            initial state (mean) for each x, y, theta. Number of particles
            M dictated in Class Initialisation.

            Args:
                self.position (mean) ~ list: initial state (x, y, theta)
                std ~ list: standard deviation (x, y, theta)

            Returns:
                numpy array of particles (M rows, 3 cols for x, y, theta)
                Gaussian Distribution
        """
        # populate x column, index 0
        self.particles[:, 0] = self.position[0] + np.random.normal(
            0, std[0], self.M)
        # populate y column, index 1
        self.particles[:, 1] = self.position[1] + np.random.normal(
            0, std[1], self.M)
        # populate theta column, index 2 and bound to 0-2pi
        self.particles[:, 2] = (self.position[2] + np.random.normal(
            0, std[2], self.M)) % (2 * np.pi)

        return self.particles

    def fwd_prop(self, control, t_next):
        """
        Propagates each particle forward using the motion model described in
        section 1.1 of the report.
        """
        dt = np.abs(control[0] - t_next)
        v_control = control[1]
        w_control = control[2]

        if w_control == 0:
            # only linear velocity
            self.particles[:, 0] += v_control * np.cos(
                self.particles[:, 2]) * dt + self.motion_noise[0]
            self.particles[:, 1] += v_control * np.sin(
                self.particles[:, 2]) * dt + self.motion_noise[1]
            # no heading update
        else:
            # lin and ang velocity, move in a circle, P-101 Probabilistic Robotics
            self.particles[:, 0] += (v_control / w_control) * (
                -np.sin(self.particles[:, 2]) +
                np.sin(self.particles[:, 2] +
                       w_control * dt)) + +self.motion_noise[0]
            self.particles[:, 1] += (v_control / w_control) * (
                np.cos(self.particles[:, 2]) -
                np.cos(self.particles[:, 2] +
                       w_control * dt)) + self.motion_noise[0]
            self.particles[:, 2] += w_control * dt + self.motion_noise[1]

    def weight(self, landmark_groundtruth, measurements):
        """
        Assign a weight to each particle according to the process described
        in section 1.4, item 3.
        """
        # Initialize list for landmarks measured
        landmarks = []

        # Make another array for the weights for m measurements to avg later
        weights = np.array([], dtype=np.float64)
        # Update becomes True if landmark (not another robot) is measured
        update = False
        for m in range(len(measurements)):
            # weights_i will be the average weight using range and bearing for
            # measurement m
            weights_i = np.array([], dtype=np.float64)
            weights_i_range = np.array([], dtype=np.float64)
            weights_i_bearing = np.array([], dtype=np.float64)
            # extract landmark attributes for each measurement
            for lg in range(len(landmark_groundtruth)):
                if landmark_groundtruth[lg][0] == measurements[m][1]:
                    update = True
                    landmarks = landmark_groundtruth[lg]
                    for i in range(self.M):
                        rb = self.measure(landmarks, i)
                        rb_mes = [measurements[m][2], measurements[m][3]]
                        # Likelihood of this particle representing the bel(x)
                        # given these measurements
                        varr = 0.5 * abs(rb[0] - rb_mes[0])
                        varb = 0.5 * abs(rb[1] - rb_mes[1])
                        prob = np.array([[(1/np.sqrt(2*np.pi*varr**2))*np.exp(-0.5*((rb[0]-rb_mes[0])**2/(varr**2)))],
                                         [(1/np.sqrt(2*np.pi*varb**2))*np.exp(-0.5*((rb[1]-rb_mes[1])**2/(varb**2)))]])
                        """prob = np.array([[1 / np.square(rb[0] - rb_mes[0])],
                                         [1 / np.square(rb[1] - rb_mes[1])]])
                        """
                        # print(prob)
                        weights_i_range = np.append(weights_i_range,
                                                    prob[0],
                                                    axis=0)
                        weights_i_bearing = np.append(weights_i_bearing,
                                                      prob[1],
                                                      axis=0)
                    # Normalise weights before averaging
                    weights_i_range /= np.sum(weights_i_range)
                    weights_i_bearing /= np.sum(weights_i_bearing)
                    weights_i = np.array([[weights_i_range],
                                          [weights_i_bearing]])
                    # Average weights for range and bearing
                    weights_i = weights_i.mean(axis=0)
            # append by column for m measurements
            if weights_i.size > 0 and weights.size == 0:
                # Initialise the weights array for the first measurement
                weights = weights_i
            elif weights_i.size and len(weights.shape) > 0:
                # Stack weight arrays for additional measurements
                np.column_stack((weights, weights_i))
        # average weight for m columns for m measurements
        if update is True:
            if len(weights.shape) > 1:
                # If weight has more than 1 column, average all
                weights = weights.mean(axis=0)
            # Weights updated multiplicatively
            self.particles[:, 3] *= weights
            # Normalise again
            self.particles[:, 3] /= np.sum(self.particles[:, 3])

    def resample(self, var):
        """
        Low-variance resampling algorithm described in section 1.4
        item 5
        """
        # Empty intermediate set chi
        X = np.empty((self.M, 4))
        # randomly generated partitioning constant
        r = np.random.random() / self.M
        # Weight of first particle
        c = self.particles[0, 3]
        # Index of particles to be sampled
        i = 0
        for m in range(self.M):
            u = r + (m * 1 / (self.M - 1))
            while u > c:
                i += 1
                if i > self.M - 1:
                    i = self.M - 1
                    break
                c += self.particles[i, 3]
            X[m, :] = self.particles[i, :]
        self.particles = X
        # re-normalise
        self.particles[:, 3] /= np.sum(self.particles[:, 3])
        # If variance below threshold, add random particles around
        # last posterior to increase variance
        if var < 3e-10:
            for i in range(self.M):
                if i % 100 == 0:
                    self.particles[i, 0] = np.random.normal(
                        self.position[0], 0.0001)
                    self.particles[i, 1] = np.random.normal(
                        self.position[1], 0.0001)
                    self.particles[i, 2] = np.random.normal(
                        self.position[2], 2)
                    self.particles[i, 3] = 1 / (self.M)
            # re-normalise
            self.particles[:, 3] /= np.sum(self.particles[:, 3])

    def posterior(self):
        """
        Extract mean and variance of posterior as reported in section 1.4
        item 7
        """
        mean = np.average(self.particles, weights=self.particles[:, 3], axis=0)
        var = np.average((self.particles - mean)**2,
                         weights=self.particles[:, 3],
                         axis=0)
        return mean, var[3]


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

    #Plot
    plt.show()

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


# Full Particle Filter with exercise 2 data
def pf2():
    # Initialize Robot instance
    position = [0, 0, 0]
    sensor_noise = [0.1, 0.1, 0.01]
    motion_noise = [0.0001, 0.0001]
    std = [0.0002, 0.0002, 1.5]
    # Number of particles
    M = 1000
    # Initialize robot instance of Robot class
    robot = Robot(position, M, sensor_noise, motion_noise)
    # Initialise particles normally distributed around starting state
    robot.init_known_particles(std)
    path = []
    # Manual transcription of controls for Exercise 2
    control = [
        position, [1, 0.5, 0], [1, 0, -1 / (2 * np.pi)], [1, 0.5, 0],
        [1, 0, 1 / (2 * np.pi)], [1, 0.5, 0]]

    # Initialize Plot
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Particle Filter Plot for Ex2 Commands')
    plt.ylabel('y [m]')
    plt.xlabel('x [m]')

    t_next = 0

    for t in range(len(control)):
        robot.fwd_prop(control[t], t_next)
        mean, var = robot.posterior()
        path.append(mean)

    # Parse F Path
    path_x = [x[0] for x in path]
    path_y = [y[1] for y in path]

    # Plot inital position
    plt.plot(path_x[0],
             path_y[0],
             color='gold',
             marker='o',
             markersize=10,
             label='Starting Point')

    # Plot final position (Particle Filter)
    plt.plot(path_x[-1],
             path_y[-1],
             color='darkviolet',
             marker='o',
             markersize=5,
             label='Endpoints')

    # Plot Path
    plt.plot(path_x, path_y, '-k', label='Particle Filter Path')

    plt.legend()
    plt.show()


# Full Particle Filter with Odometry Data
def pf(odometry, ground_truth, measurement, barcodes, landmark_groundtruth):
    # Convert Measurement.dat subject # into Landmark subject # w barcode.dat
    for b in range(len(barcodes)):
        for mes in range(len(measurement)):
            if barcodes[b][1] == measurement[mes][1]:
                measurement[mes][1] = barcodes[b][0]

    # Initial position at ground truth
    position = [1.29812900, 1.88315210, 2.82870000]
    sensor_noise = [0.1, 0.1, 0.01]
    motion_noise = [0.0001, 0.0001]
    std = [0.0002, 0.0002, 1.5]
    # Number of particles
    M = 1000
    # Initialize robot instance of Robot class
    robot = Robot(position, M, sensor_noise, motion_noise)
    # Initialise particles normally distributed around starting state
    robot.init_known_particles(std)

    # Initialise Plot
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title('Particle Filter Pose Estimation VS. Ground Truth Data')
    plt.ylabel('y [m]')
    plt.xlabel('x [m]')

    # Plot particles
    plt.scatter(robot.particles[:, 0],
                robot.particles[:, 1],
                alpha=0.2,
                color='darkviolet',
                label=' Initial Particles')
    # Initialise path
    path = []
    var = 0
    # Loop for all odometry commands
    iterations = len(odometry) - 1
    for t in range(iterations):
        t_next = odometry[t + 1][0]
        t_current = odometry[t][0]
        robot.fwd_prop(odometry[t], t_next)

        # Initialize measurements list used in this t
        measurements = []
        # If landmark timestamp within two control stamps (future-curr), use it
        for m in range(len(measurement)):
            if measurement[m][0] >= t_current and measurement[m][0] <= t_next:
                robot.last_measurement = m - 10  # pick up from here next time
                # only track landmarks, not other robots
                if measurement[m][1] >= 6 and measurement[m][1] <= 20:
                    measurements.append(measurement[m])
        # If measurement list contains measurement
        if len(measurements) > 0:
            print("measure at t = {} of 95812".format(t))
            # Weigh particles using measurement
            robot.weight(landmark_groundtruth, measurements)
        # If var is too high, weight is too concentrated
        # so resample
        if var > 4e-10:
            # print("resample")
            robot.resample(var)

        # Extract posterior and var
        mean, var = robot.posterior()
        path.append(mean)
        robot.position = mean[0:3]

    ##################### PLOTTING ###############################

    # Parse F Path
    path_x = [x[0] for x in path]
    path_y = [y[1] for y in path]

    # Parse Ground Truth Path
    ground_truth_x = [x[1] for x in ground_truth]
    ground_truth_y = [y[2] for y in ground_truth]
    """
    ground_truth_xs = []
    ground_truth_ys = []
    for gx in range(iterations):
        ground_truth_xs.append(ground_truth_x[gx])
    for gy in range(iterations):
        ground_truth_ys.append(ground_truth_y[gy])
    """

    plt.plot(path_x, path_y, '-k', label='Particle Filter Path')
    plt.plot(ground_truth_x, ground_truth_y, '-g', label='Ground Truth Data')

    # Plot inital position (Both)
    plt.plot(path_x[0],
             path_y[0],
             color='gold',
             marker='o',
             markersize=10,
             label='Starting Point')

    # Plot final position (Particle Filter)
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

    # Plot Landmarks
    landmark_x = [x[1] for x in landmark_groundtruth]
    landmark_y = [y[2] for y in landmark_groundtruth]
    plt.scatter(landmark_x,
                landmark_y,
                color='r',
                marker='x',
                label='Landmarks')
    # plt.legend()
    plt.show()


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
    measurement = read_dat(
        3, "ds0/ds0_Measurement.dat",
        ["Time [s]", "Subject #", "range [m]", "bearing [rad]"])
    barcodes = read_dat(3, "ds0/ds0_Barcodes.dat", ["Subject #", "Barcode #"])

    # Select Exercise
    exercise = raw_input('Select an exercise [2,3,6,7]')
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
    elif exercise == '7':
        # Particle Filter
        plot_set = raw_input('Plot Particle Filter for exercise 2 or 3? [2,3]')
        if plot_set == '2':
            pf2()
        elif plot_set == '3':
            pf(odometry, ground_truth, measurement, barcodes,
               landmark_groundtruth)


if __name__ == "__main__":
    main()
