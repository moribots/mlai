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
import scipy as sp
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

    def measure(self, landmark):
        """
        FILL DOCSTRING
        """
        # Retrieve range/bearing for a particle based on known measured landmark and its std
        mu = 0
        sigma_x = landmark[3]
        sigma_y = landmark[4]
        cov_x = (np.random.normal(mu, sigma_x))**2
        cov_y = (np.random.normal(mu, sigma_y))**2
        r2l_range = np.zeros(self.M)
        r2l_range[:] = np.sqrt(
            np.power(self.particles[:, 0] - landmark[1] - cov_x, 2) +
            np.power(self.particles[:, 1] - landmark[1] - cov_y, 2))
        # arctan2 has built-in logic to account for quadrants
        r2l_bearing = np.zeros(self.M)
        r2l_bearing[:] = np.arctan2(
            (landmark[2] + cov_y - self.particles[:, 1]),
            (landmark[1] + cov_x - self.particles[:, 0]))
        rb = [r2l_range, r2l_bearing]
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
        # Initialize rb with range, bearing columns
        rb = np.empty(self.M, 2)
        # Initialize dist_weights as intermediate weight array
        # to be averaged later for each landmark result
        dist_weights = np.ones(self.M, 1)
        # Similarly for bearings weight
        bear_weights = np.ones(self.M, 1)
        # Make another array for the weights for m measurements to avg later
        weights = []
        # For each measurement on valid timestep, record
        # measured landmark attributes and compare
        # predicted range/bearing of each particle wrt it
        for m in range(len(measurements)):
            # extract landmark attributes for each measurement
            for lg in range(len(landmark_groundtruth)):
                if landmark_groundtruth[lg][0] == measurements[lg][1]:
                    landmarks.append(landmark_groundtruth[lg])
            # compute range, bearing for each particles
            rb = self.measure(landmarks[m])
            # calculate weight of each particle for landmark m - distance
            dist_weights[:, 0] = np.exp(
                -((rb[:, 0] - measurements[2])**2) /
                (self.sensor_noise**2) / 2.0) / np.sqrt(2.0 * np.pi *
                                                        (self.sensor_noise**2))
            bear_weights[:, 0] = np.exp(
                -((rb[:, 1] - measurements[3])**2) /
                (self.sensor_noise**2) / 2.0) / np.sqrt(2.0 * np.pi *
                                                        (self.sensor_noise**2))
            # take average across rows and store for each measurement m
            weights.append(
                np.mean(np.array([dist_weights, bear_weights]), axis=0))
        # average weight for m columns for m measurements
        self.particles[:, 3] = weights.mean(axis=1)


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
    sensor_noise = 0.00017939
    motion_noise = 0.00017939
    robot = Robot(position, 10, sensor_noise, motion_noise)
    std = [0.00017939, 0.00017939, 0.00017939]
    robot.init_known_particles(std)
    print(robot.particles.shape)

if __name__ == "__main__":
    main()
