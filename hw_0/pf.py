#!/usr/bin/env python
"""
Python 2.7 PEP8 Style
Code submission for Homework 0 Part B
Machine Learning & Artificial Intelligence for Robotics
Data Set 0
Particle Filter Implementation

Maurice Rahme
mauricerahme2020@u.northwestern.edu
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
                        """
                        varr = 0.5 * abs(rb[0] - rb_mes[0])
                        varb = 0.5 * abs(rb[1] - rb_mes[1])
                        prob = np.array([[(1/np.sqrt(2*np.pi*varr**2))*np.exp(-0.5*((rb[0]-rb_mes[0])**2/(varr**2)))],
                                         [(1/np.sqrt(2*np.pi*varb**2))*np.exp(-0.5*((rb[1]-rb_mes[1])**2/(varb**2)))]])
                        """
                        prob = np.array([[1 / np.square(rb[0] - rb_mes[0])],
                                         [1 / np.square(rb[1] - rb_mes[1])]])
                        
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