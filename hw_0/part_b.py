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
        self.particles[:, 3] = 1 / self.M
        # Initialize sensor noise
        self.sensor_noise = sensor_noise
        # Initialize motion noise
        self.motion_noise = motion_noise
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
        # arctan2 has built-in logic to account for quadrants
        r2l_bearing = np.zeros((self.M, 1))
        r2l_bearing = np.arctan2((landmark[2] + cov_y - self.particles[i, 1]),
                                 (landmark[1] + cov_x - self.particles[i, 0]))
        rb = np.array([r2l_range, r2l_bearing])
        return rb

    def init_unknown_particles(self, x_domain, y_domain, bearing_range):
        """ USE IF INITIAL CONDITION (STATE) UNKNOWN

            Generates random uniformly distributed particles, each with
            x, y, and theta attributes. Number of particles dictated in
            Class initialisation.

            Args:
                x_domain ~ list: room size in x
                y_domain ~ list: room size in y
                bearing_range ~ list: should be set from 0 - 2*pi
                last arg modulated by 2*np.pi for bounding assurance

            Returns:
                numpy array of particles (M rows, 3 cols for x, y, theta)
                Uniform Distribution
        """
        # populate x column, index 0
        self.particles[:, 0] = np.random.uniform(x_domain[0],
                                                 x_domain[1],
                                                 size=self.M)
        # populate y column, index 1
        self.particles[:, 1] = np.random.uniform(y_domain[0],
                                                 y_domain[1],
                                                 size=self.M)
        # populate theta column and take modulus of 2pi to keep in bounds
        self.particles[:, 2] = np.random.uniform(
            bearing_range[0], bearing_range[1], size=self.M) % 2 * np.pi

        return self.particles

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
        self.particles[:, 2] = (
            self.position[2] + np.random.normal(0, std[2], self.M)) % 2 * np.pi

        return self.particles

    def fwd_prop(self, control, t_next):
        """
        FILL DOCSTRING
        """
        dt = np.abs(control[0] - t_next)

        # std[0] is linear control noise
        v_control = control[1] * (1 + self.motion_noise)
        # std[1] is angular control noise
        w_control = control[2] * (1 + self.motion_noise)

        if w_control == 0:
            # only linear velocity
            self.particles[:, 0] += self.particles[:, 0] * control[1] * dt
            self.particles[:, 1] += self.particles[:, 1] * control[1] * dt
            # no heading update
        else:
            # lin and ang velocity, move in a circle, P-101 PR
            self.particles[:, 0] += (v_control / w_control) * (
                -np.sin(self.particles[:, 2]) +
                np.sin(self.particles[:, 2] + w_control * dt))
            self.particles[:, 1] += (v_control / w_control) * (
                np.cos(self.particles[:, 2]) -
                np.cos(self.particles[:, 2] + w_control * dt))
            self.particles[:, 2] += w_control * dt

    def weight(self, landmark_groundtruth, measurements):
        """
        FILL DOCSTRING
        """
        # Initialize list for landmarks we measured
        landmarks = []
        # Initialize dist_weights as intermediate weight array
        # to be averaged later for each landmark result

        # Make another array for the weights for m measurements to avg later
        weights = np.array([])
        # For each measurement on valid timestep, record
        # measured landmark attributes and compare
        # predicted range/bearing of each particle wrt it
        update = False
        for m in range(len(measurements)):
            weights_i = np.array([])
            # extract landmark attributes for each measurement
            for lg in range(len(landmark_groundtruth)):
                if landmark_groundtruth[lg][0] == measurements[m][1]:
                    landmarks = landmark_groundtruth[lg]
                    for i in range(self.M):
                        rb = self.measure(landmarks, i)
                        rb_mes = np.array(
                            [measurements[m][2], measurements[m][3]])
                        prob = np.array([
                            scipy.stats.norm(rb, self.sensor_noise).pdf(rb_mes)
                        ])
                        # print("prob{}".format(prob))
                        prob = prob.mean(axis=1)
                        print("prob{}".format(prob))
                        """
                        cov_matrix = np.array(
                            [[
                                self.motion_noise**2,
                                self.motion_noise * self.sensor_noise
                            ],
                             [
                                 self.motion_noise * self.sensor_noise,
                                 self.sensor_noise**2
                             ]])
                        prob = np.array([
                            np.linalg.det(2 * np.pi * cov_matrix)**(-0.5) *
                            np.exp(-0.5 * np.transpose(rb - rb_mes) *
                                   np.linalg.inv(cov_matrix) * (rb - rb_mes))
                        ])
                        """

                        # prob = np.array([0.5])
                        # axis = 0 is row append
                        weights_i = np.append(weights_i, prob, axis=0)
                    update = True
            # append by column for m measurements
            if weights_i.size > 0 and weights.size == 0:
                weights = weights_i
            elif weights_i.size > 0:
                # np.column_stack((weights, weights_i))
                weights = np.append(weights, weights_i, axis=1)
        # average weight for m columns for m measurements
        if update is True:
            if len(weights.shape) > 1:
                weights = weights.mean(axis=1)  # average weights for each mes
            weights += 1e-100  # avoid round-off to zero
            weights /= sum(weights)  # normalise
            print(sum(weights))
            self.particles[:, 3] = weights

    def neff(self):
        """
        """
        return 1. / np.sum(self.particles[:, 3]**2)

    def resample(self):
        """
        """
        X = np.array()
        r = np.random.uniform(0, (1 / self.M))
        c = self.particles[0, 3]  # first weight element
        i = 1
        for m in range(1, self.M):
            u = r + (m - 1) * (1 / self.M)
            while u > c:
                i += 1
                c += self.particles[i, 3]
            X = np.append(X, self.particles[i, :], axis=0)
        self.particles[:, 3] = X

    def posterior(self):
        mean = np.average(self.particles[:, 0:2],
                          weights=self.particles[:, 3],
                          axis=0)
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

    position = [1.29812900, 1.88315210,
                2.82870000]  # Set equal to Ground Truth Initial
    sensor_noise = 0.3
    motion_noise = 0.17939
    std = [0.17939, 0.17939, 0.17939]
    M = 3
    # Initialize robot instance of Robot class
    robot = Robot(position, M, sensor_noise, motion_noise)
    # Initialise particles normally distributed around starting state
    robot.init_known_particles(std)

    path = []
    # Loop for all odometry commands
    # REPLACE RANGE WITH 0 --> LEN CONTROLS -1
    for t in range(388, 500):
        t_next = odometry[t + 1][0]
        t_current = odometry[t][0]
        robot.fwd_prop(odometry[t], t_next)
        # path.append(robot.particles.tolist())

        # Initialize measurements list used in this t
        measurements = []
        # If landmark timestamp within two control stamps (future-curr), use it
        # look through measurements to see what timestamps match
        for m in range(robot.last_measurement, len(measurement)):
            if measurement[m][0] >= t_current and measurement[m][0] <= t_next:
                robot.last_measurement = m  # pick up from here next time
                # only track landmarks, not other robots
                if measurement[m][1] >= 6 and measurement[m][1] <= 20:
                    measurements.append(measurement[m])
        if len(measurements) > 0:
            robot.weight(landmark_groundtruth, measurements)
            # print(robot.particles)


if __name__ == "__main__":
    main()
