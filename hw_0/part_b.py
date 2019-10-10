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

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from __future__ import division, print_function


# Robot class contains position attributes (x, y, theta) and move method
class Robot():
    def __init__(self, position, num_particles):
        self.position = position  # x, y, theta at world origin
        self.M = num_particles  # number of particles used in filter
        # empty numpy array w/N rows, 3 columns for x,y,theta
        self.particles = np.empty((self.M, 3))
        # set initial particle weights to be all equal normalised to sum to 1
        self.weights = np.ones((self.M, 3)) * 1 / self.M

    def measure(self, position, landmark):
        """
        FILL DOCSTRING
        """
        # Retrieve range/bearing for each particle based on known measured landmark and its std
        sigma_x = landmark[3]
        sigma_y = landmark[4]
        cov_x = (np.random.normal(mu, sigma_x))**2
        cov_y = (np.random.normal(mu, sigma_y))**2

        r2l_range = np.sqrt((position[0] - landmark[1] - cov_x)**2 +
                            (position[1] - landmark[2] - cov_y)**2)
        # arctan2 has built-in logic to account for quadrants
        r2l_bearing = np.arctan2((landmark[2] + cov_y - position[1]),
                                 (landmark[1] + cov_x - position[0]))
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

    def init_known_particles(self, mean, std):
        """ USE IF INITIAL CONDITION (STATE) KNOWN

            Generates a gaussian distribution of particles about the
            initial state (mean) for each x, y, theta. Number of particles
            dictated in Class Initialisation.

            Args:
                mean ~ list: initial state (x, y, theta)
                std ~ list: standard deviation (x, y, theta)

            Returns:
                numpy array of particles (M rows, 3 cols for x, y, theta)
                Gaussian Distribution
        """
        # populate x column, index 0
        self.particles[:, 0] = mean[0] + np.random.normal(0, std[0], self.M)
        # populate y column, index 1
        self.particles[:, 1] = mean[1] + np.random.normal(0, std[1], self.M)
        # populate theta column, index 2 and bound to 0-2pi
        self.particles[:, 2] = (
            mean[2] + np.random.normal(0, std[2], self.M)) % 2 * np.pi

        return self.particles

    def fwd_prop(self, control, t_next, std):
        """
        FILL DOCSTRING
        """
        dt = np.abs(control[0] - t_next)

        # std[0] is linear control noise
        v_control = control[1] * (1 + std[0])
        # std[1] is angular control noise
        w_control = control[2] * (1 + std[1])

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

    def weight(self, landmark_groundtruth, measurement):
        """
        FILL DOCSTRING
        """
        # measure what range and bearing should be in each particle, and compare to actual measured range/bearing





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


if __name__ == "__main__":
    main()
