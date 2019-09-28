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

    for t in range(len(control)):
        # Method in Robot class which updates pose based on controls
        robot.move(control[t], 0)  # No dt_prev here since dt is defined
        #print(robot.position)
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
    control = odometry

    for t in range(len(control)):
        t_next = control[t + 1][0]

        robot.move(control[t], t_next)
        #print(robot.position)
        path.append(robot.position)
        plt.title('Motion Iteration: ' + str(t - 1))
        # Plot first position clearly
        if t == 0:
            plt.plot(robot.position[0], robot.position[1], 'ko', markersize=10)
    # Plot other positions usind lines and diamonds
        else:
            plt.plot([path[t - 1][0], path[t][0]],
                     [path[t - 1][1], path[t][1]], '-k')
    # Realtime plot update
        plt.pause(0.00001)
    return robot


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
    # Turn string data into floats
    data = []
    for i in range(start_index, len(data_str)):
        data.append(np.array(data_str[i], dtype=np.float64))

    return data


# Main

# Exercise 2
plt.xlim((-0.2, 2))
plt.ylim((-0.5, 0.2))
plt.title('Motion Iteration: 0')
plt.ylabel('y domain')
plt.xlabel('x domain')
robot = a2()

# Plot final position clearly
plt.plot(robot.position[0], robot.position[1], color='darkviolet', marker='o', markersize=10)

"""
# Exercise 3
plt.xlim((-2, 2))
plt.ylim((-2, 2))
plt.title('Motion Iteration: 0')
plt.ylabel('y domain')
plt.xlabel('x domain')

odometry = read_dat(
    3, "ds0/ds0_Odometry.dat",
    ["Time [s]", "forward velocity [m/s]", "angular velocity[rad/s]"])

robot = a3(odometry)

# Plot final position clearly
plt.plot(robot.position[0], robot.position[1], 'po', markersize=10)
"""

plt.show()
