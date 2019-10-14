#!/usr/bin/env python
"""
Python 2.7 PEP8 Style
Code submission for Homework 0 Part B
Machine Learning & Artificial Intelligence for Robotics
Data Set 0
Particle Filter Implementation

Data interpreted with Python's Pandas library

Maurice Rahme
Student ID: 3219435
mauricerahme2020@u.northwestern.edu
"""

from __future__ import division
import numpy as np
import math
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd


# Robot class contains position attributes (x, y, theta) and move method
class Robot():
    def __init__(self, position, num_particles, sensor_noise, motion_noise):
        self.position = position  # x, y, theta at world origin
        self.M = num_particles  # number of particles used in filter
        # empty numpy array w/N rows, 4 columns for x,y,theta, weight
        self.particles = np.zeros((self.M, 4))
        # set initial particle weights (4th col) to be all equal and sum to 1
        self.particles[:, 3] = 1.0 / self.M
        # Initialize sensor noise
        self.sensor_noise = [np.random.normal(0, sensor_noise[0]), np.random.normal(0, sensor_noise[1]), np.random.normal(0, sensor_noise[2])]
        # Initialize motion noise
        self.motion_noise = [np.random.normal(0, motion_noise[0]), np.random.normal(0, motion_noise[1])]
        # Record last measurement to avoid reading list from scratch
        self.last_measurement = 0

    def measure(self, landmark, i):
        """
        FILL DOCSTRING
        """
        # Retrieve range/bearing for a particle based on known measured landmark and its std
        mu = 0
        sigma_x = landmark[3]
        sigma_y = landmark[4]
        cov_x = (np.random.normal(mu, sigma_x))**2
        cov_y = (np.random.normal(mu, sigma_y))**2
        r2l_range = np.sqrt(
            np.power(self.particles[i, 0] - landmark[1] - cov_x, 2) +
            np.power(self.particles[i, 1] - landmark[2] - cov_y, 2))
        # print("for {}, range is {}".format(i, r2l_range))
        # print("for {}, x is {} and y is {}".format(i, self.particles[i, 0], self.particles[i, 1]))
        # arctan2 has built-in logic to account for quadrants
        r2l_bearing = np.arctan2((landmark[2] + cov_y - self.particles[i, 1]),
                                 (landmark[1] + cov_x - self.particles[i, 0]))
        rb = [r2l_range, r2l_bearing]
        return rb

    def init_known_particles(self, std):
        """ USE IF INITIAL CONDITION (STATE) KNOWN

            Generates a gaussian distribution of particles about the
            initial state (mean) for each x, y, theta. Number of particles
            dictated in Class Initialisation.

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
        FILL DOCSTRING
        """
        dt = np.abs(control[0] - t_next)

        # std[0] is linear control noise
        v_control = control[1]
        # std[1] is angular control noise
        w_control = control[2]

        if w_control == 0:
            # only linear velocity
            self.particles[:, 0] += v_control * np.cos(self.particles[:, 2]) * dt + self.motion_noise[0]
            self.particles[:, 1] += v_control * np.sin(self.particles[:, 2]) * dt + self.motion_noise[1]
            # no heading update
        else:
            # lin and ang velocity, move in a circle, P-101 PR
            self.particles[:, 0] += (v_control / w_control) * (
                -np.sin(self.particles[:, 2]) +
                np.sin(self.particles[:, 2] + w_control * dt)) + + self.motion_noise[0]
            self.particles[:, 1] += (v_control / w_control) * (
                np.cos(self.particles[:, 2]) -
                np.cos(self.particles[:, 2] + w_control * dt)) + self.motion_noise[0]
            self.particles[:, 2] += w_control * dt + self.motion_noise[1]

    def weight(self, landmark_groundtruth, measurements):
        """
        FILL DOCSTRING
        """
        # Initialize list for landmarks we measured
        landmarks = []
        # Initialize dist_weights as intermediate weight array
        # to be averaged later for each landmark result

        # Make another array for the weights for m measurements to avg later
        weights = np.array([], dtype=np.float64)
        # For each measurement on valid timestep, record
        # measured landmark attributes and compare
        # predicted range/bearing of each particle wrt it
        update = False
        for m in range(len(measurements)):
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
                        # print("difference: {}".format(rb[0] - rb_mes[0]))
                        # prob = np.array([[scipy.stats.norm(rb[0], self.sensor_noise).pdf(rb_mes[0])], [scipy.stats.norm(rb[1], self.sensor_noise).pdf(rb_mes[1])]])
                        # Smaller distance should warrant higher weight
                        prob = np.array([[1 / np.square(rb[0] - rb_mes[0])],
                                         [1 / np.square(rb[1] - rb_mes[1])]])
                        # print(prob)
                        weights_i_range = np.append(weights_i_range,
                                                    prob[0],
                                                    axis=0)
                        weights_i_bearing = np.append(weights_i_bearing,
                                                      prob[1],
                                                      axis=0)
                    # weights_i_range += 1e-300
                    # weights_i_bearing += 1e-300
                    weights_i_range /= np.sum(weights_i_range)
                    weights_i_bearing /= np.sum(weights_i_bearing)
                    # print("range w {}".format(weights_i_range))
                    # print("bear w {}".format(weights_i_bearing))
                    weights_i = np.array([[weights_i_range],
                                          [weights_i_bearing]])
                    weights_i = weights_i.mean(axis=0)
            # append by column for m measurements
            if weights_i.size > 0 and weights.size == 0:
                weights = weights_i
            elif weights_i.size and len(weights.shape) > 0:
                np.column_stack((weights, weights_i))
        # average weight for m columns for m measurements
        if update is True:
            if len(weights.shape) > 1:
                weights = weights.mean(axis=0)
            self.particles[:, 3] *= weights
            self.particles[:, 3] /= np.sum(self.particles[:, 3])

    def eff_weights(self):
        """
        """
        w_eff = 1.0 / np.sum(self.particles[:, 3]**2)
        return w_eff

    def resample(self):
        """
        """
        X = np.empty((self.M, 4))
        r = np.random.random() * 1 / float(self.M)
        c = self.particles[0, 3]  # first weight element
        i = 0
        for m in range(self.M):
            u = r + (m * 1 / float(self.M - 1))
            while u > c:
                i += 1
                if i > self.M - 1:
                    i = self.M - 1
                    break
                c += self.particles[i, 3]
            X = np.append(X, self.particles[i, :])
        self.particles = X

    def lowvar_resample(self):
        index_array = (np.arange(self.M) +
                       np.random.uniform(0, (1.0 / self.M))) / self.M
        # print(index_array)

        indxs = np.zeros(self.M, dtype=np.int32)
        cum_sum = np.cumsum(self.particles[:, 3])
        i = 0
        j = 0
        while i < self.M:
            if index_array[i] < cum_sum[j]:
                indxs[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[indxs, :]
        # re-normalise
        self.particles[:, 3] /= np.sum(self.particles[:, 3])
        if self.eff_weights() > self.M * 0.85:
            # resampling needs additional variance
            for i in range(self.M):
                if i % 100 == 0:
                    self.particles[i, 0] = np.random.normal(
                        self.position[0], 0.0001)
                    self.particles[i, 1] = np.random.normal(
                        self.position[1], 0.0001)
                    self.particles[i, 2] = np.random.normal(self.position[2], 0.05)
                    self.particles[i, 3] = 1 / float(self.M)
            # re-normalise
            self.particles[:, 3] /= np.sum(self.particles[:, 3])

    def posterior(self):
        """
        """
        # mean = np.average(self.particles, weights=self.particles[:, 3], axis=0)
        #print(mean)
        
        mean_x = 0
        mean_y = 0
        mean_t = 0
        for i in range(self.M):

            mean_x += self.particles[i, 0] * self.particles[i, 3]
            mean_y += self.particles[i, 1] * self.particles[i, 3]
            mean_t += self.particles[i, 2] * self.particles[i, 3]
        mean = np.array([mean_x, mean_y, mean_t])
        return mean


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
    measurement = read_dat(
        3, "ds0/ds0_Measurement.dat",
        ["Time [s]", "Subject #", "range [m]", "bearing [rad]"])
    barcodes = read_dat(
        3, "ds0/ds0_Barcodes.dat",
        ["Subject #", "Barcode #"])

    # Convert Measurement.dat subject # into actual subject # for landmark_groundtruth using barcode.dat
    for b in range(len(barcodes)):
        for mes in range(len(measurement)):
            if barcodes[b][1] == measurement[mes][1]:
                measurement[mes][1] = barcodes[b][0]
    # for mi in range(len(measurement)):
    #    print(measurement[mi][1])



    position = [1.29812900, 1.88315210, 2.82870000]  # Set equal to Ground Truth Initial
    sensor_noise = [0.001, 0.001, 0.001]
    motion_noise = [0.0001, 0.0001]  # 10% noise on controller
    std = [0.2, 0.2, 0.5]
    M = 1000
    # Initialize robot instance of Robot class
    robot = Robot(position, M, sensor_noise, motion_noise)
    # Initialise particles normally distributed around starting state
    robot.init_known_particles(std)
    print(robot.particles)
    print(robot.eff_weights())

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
                label='Particles')

    path = []
    resample = 0
    # Loop for all odometry commands
    # REPLACE RANGE WITH 0 --> LEN CONTROLS -1
    iterations = 10000
    for t in range(iterations):
        t_next = odometry[t + 1][0]
        t_current = odometry[t][0]
        robot.fwd_prop(odometry[t], t_next)
        # path.append(robot.particles.tolist())

        # Initialize measurements list used in this t
        measurements = []
        # If landmark timestamp within two control stamps (future-curr), use it
        # look through measurements to see what timestamps match
        for m in range(len(measurement)):
            if measurement[m][0] >= t_current and measurement[m][0] <= t_next:
                robot.last_measurement = m  # pick up from here next time
                # only track landmarks, not other robots
                if measurement[m][1] >= 6 and measurement[m][1] <= 20:
                    measurements.append(measurement[m])
        if len(measurements) > 0:
            print("measure at t = {}".format(t))
            robot.weight(landmark_groundtruth, measurements)

        # print(robot.eff_weights())
        # robot.eff_weights() < robot.M * 0.99
        # Resample if Neff < N/2 (not enough high weight particles)
        if len(measurements) > 0 or robot.eff_weights() < robot.M/2:
        # if robot.eff_weights() < robot.M * 0.89:
            print("resample")
            robot.lowvar_resample()
            resample += 1
        else:
            resample += 1
        # elif robot.eff_weights() > robot.M * 0.9:
        # Our particle set has collapsed to one item, rebuild distro

        # Extract posterior
        mean = robot.posterior()
        path.append(mean)
        robot.position = mean[0:3]
        # print("the mean is: {}".format(mean))

    # Parse F Path
    path_x = [x[0] for x in path]
    path_y = [y[1] for y in path]

    # Parse Ground Truth Path
    ground_truth_x = [x[1] for x in ground_truth]
    ground_truth_y = [y[2] for y in ground_truth]
    ground_truth_xs = []
    ground_truth_ys = []
    for gx in range(iterations):
        ground_truth_xs.append(ground_truth_x[gx])
    for gy in range(iterations):
        ground_truth_ys.append(ground_truth_y[gy])
    

    plt.plot(path_x, path_y, '-k', label='Particle Filter Path')
    plt.plot(ground_truth_xs, ground_truth_ys, '-g', label='Ground Truth Data')

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
    plt.plot(ground_truth_xs[-1],
             ground_truth_ys[-1],
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

    # Plot particles
    plt.scatter(robot.particles[:, 0],
                robot.particles[:, 1],
                alpha=0.2,
                color='r',
                label='Particles')
    # print(len(robot.particles))
    #plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
