#!/usr/bin/env python
# use chmod +x file.py (+x grants executable permission)
# use ./file.py to run
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

    def move(self, control):
        x = self.position[0] + control[0] * np.cos(
            self.position[2]) * control[2]
        y = self.position[1] + control[0] * np.sin(
            self.position[2]) * control[2]
        theta = self.position[2] + control[1] * control[2]
        self.position = [x, y, theta]


# Exercise 2, plot robot movement using given commands
def a2():
    # Initialize Robot instance
    position = [0, 0, 0]
    robot = Robot(position)
    path = []
    # Manual transcription of controls for Exercise 2
    control = [
        position, [0.5, 0, 1], [0, -1 / (2 * np.pi), 1], [0.5, 0, 1],
        [0, 1 / (2 * np.pi), 1], [0.5, 0, 1]
    ]

    for t in range(len(control)):
        # Method in Robot class which updates pose based on controls
        robot.move(control[t])
        print(robot.position)
        path.append(robot.position)
        plt.title('Motion Iteration: ' + str(t - 1))
        if t == 0:
            plt.plot(robot.position[0], robot.position[1], 'ko', markersize=10)
        else:
            plt.plot([path[t - 1][0], path[t][0]],
                     [path[t - 1][1], path[t][1]], '-kd')
        plt.pause(0.05)
    return robot


# Exercise 3, plot robot movement using Odometry.dat
def a3(odometry):
    position = [0, 0, 0]
    robot = Robot(position)
    path = []
    control = [
        position, [0.5, 0, 1], [0, -1 / (2 * np.pi), 1], [0.5, 0, 1],
        [0, 1 / (2 * np.pi), 1], [0.5, 0, 1]
    ]

    for t in range(len(control)):

        robot.move(control[t])
        print(robot.position)
        path.append(robot.position)
        plt.title('Motion Iteration: ' + str(t - 1))
        # Plot first position clearly
        if t == 0:
            plt.plot(robot.position[0], robot.position[1], 'ko', markersize=10)
        # Plot other positions usind lines and diamonds
        else:
            plt.plot([path[t - 1][0], path[t][0]],
                     [path[t - 1][1], path[t][1]], '-kd')
        # Realtime plot update
        plt.pause(0.05)
    return robot


# Main
plt.xlim((-0.2, 2))
plt.ylim((-0.5, 0.2))
plt.title('Motion Iteration: 0')
plt.ylabel('y domain')
plt.xlabel('x domain')

# Read Odomtry Data using Pandas
odometry_str = pd.read_table(
    "ds0/ds0_Odometry.dat",
    sep="\s+",
    skiprows=1,
    usecols=["Time [s]", "forward velocity [m/s]", "angular velocity[rad/s]"],
    names=["Time [s]", "forward velocity [m/s]", "angular velocity[rad/s]"])
odometry_str = odometry_str.values.tolist()

# Useful data starts on index 3, preceeded by headings
# Turn string data into floats
# 67Hz data recording, dt = 1/67s
odometry = []
for i in range(3, len(odometry_str)):
    odometry.append(np.array(odometry_str[i], dtype=np.float32))

# Exercise 2
robot = a2()

# Exercise 3
#robot = a3(odometry)

# Plot final position clearly
plt.plot(robot.position[0], robot.position[1], 'go', markersize=10)
plt.show()
