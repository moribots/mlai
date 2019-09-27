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


class Robot():
    def __init__(self, position):
        # Each node has one parent, but can have multiple children
        self.position = position  # x, y, theta at world origin

    def move(self, control):
        x = self.position[0] + control[0] * np.cos(
            self.position[2]) * control[2]
        y = self.position[1] + control[0] * np.sin(
            self.position[2]) * control[2]
        theta = self.position[2] + control[1] * control[2]
        self.position = [x, y, theta]


def a2():
    position = [0, 0, 0]
    robot = Robot(position)
    path = []
    # Exercise 2 in part A of Homework
    control = [
        position, [0.5, 0, 1], [0, -1 / (2 * np.pi), 1], [0.5, 0, 1],
        [0, 1 / (2 * np.pi), 1], [0.5, 0, 1]
    ]

    for t in range(len(control)):

        robot.move(control[t])
        path.append(robot.position)
        print(robot.position)
        plt.title('Motion Iteration: ' + str(t - 1))
        if t == 0:
            plt.plot(robot.position[0], robot.position[1], 'ko', markersize=10)
        else:
            plt.plot([path[t - 1][0], path[t][0]],
                     [path[t - 1][1], path[t][1]], '-kd')
        plt.pause(0.05)
    return robot


# Main
plt.xlim((-0.2, 2))
plt.ylim((-0.5, 0.2))
plt.title('Motion Iteration: 0')
plt.ylabel('y domain')
plt.xlabel('x domain')

robot = a2()

plt.plot(robot.position[0], robot.position[1], 'go', markersize=10)
plt.show()
