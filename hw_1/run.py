#!/usr/bin/env python
"""
Python 2.7 PEP8 Style
Code submission for Homework 1 Part A
Machine Learning & Artificial Intelligence for Robotics
Data Set 1
A* Search Implementation

Data interpreted with Python's Pandas library

Maurice Rahme
mauricerahme2020@u.northwestern.edu
"""

from __future__ import division
import numpy as np
from pprint import pprint
import heapq
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import pandas as pd


# Grid class
class Grid():
    """DOCSTRING
    """
    def __init__(self, cell_size, landmarks):
        self.xmin, self.xmax = -2, 5
        self.ymin, self.ymax = -6, 6
        self.cell_size = cell_size
        self.cell_xmin = np.arange(self.xmin, self.xmax, self.cell_size)
        self.cell_ymin = np.arange(self.ymin, self.ymax, self.cell_size)
        self.centres = np.ones((len(self.cell_xmin), len(self.cell_ymin)))
        self.landmarks = landmarks
        self.prev_x = 0
        self.prev_y = 0
        self.obstacles()
        if self.cell_size < 0.3:
            self.inflate_obstacles()

    def inflate_obstacles(self):
        indxs = np.where(self.centres == 1000)
        coords = list(zip(indxs[0], indxs[1]))

        for c in range(len(coords)):
            # Diagonal add
            self.centres[coords[c][0] + 1, coords[c][1] + 1] = 1000
            self.centres[coords[c][0] - 1, coords[c][1] - 1] = 1000
            self.centres[coords[c][0] + 1, coords[c][1] - 1] = 1000
            self.centres[coords[c][0] - 1, coords[c][1] + 1] = 1000

            # Horiz add
            self.centres[coords[c][0] + 1, coords[c][1]] = 1000
            self.centres[coords[c][0] - 1, coords[c][1]] = 1000

            # Vert add
            self.centres[coords[c][0], coords[c][1] + 1] = 1000
            self.centres[coords[c][0], coords[c][1] - 1] = 1000

    def obstacles(self):
        """DOCSTRING
        """
        for ob in range(len(self.landmarks)):
            index_x = []
            index_y = []
            for x in range(len(self.cell_xmin)):
                # print("x is {}".format(x))
                if self.landmarks[ob][0] >= self.cell_xmin[
                        x] and self.landmarks[ob][
                            0] <= self.cell_xmin[x] + self.cell_size:
                    index_x.append(x)

            for y in range(len(self.cell_ymin)):
                if self.landmarks[ob][1] >= self.cell_ymin[
                        y] and self.landmarks[ob][
                            1] <= self.cell_ymin[y] + self.cell_size:
                    index_y.append(y)

            if len(index_x) == 1 and len(index_y) == 1:
                self.centres[index_x, index_y] = 1000  # cost of obstacle
            elif len(index_x) > len(index_y):
                for indx in range(len(index_x)):
                    for indy in range(len(index_y)):
                        self.centres[index_x[indx], index_y[indy]] = 1000
            elif len(index_y) > len(index_x):
                for indx in range(len(index_y)):
                    for indy in range(len(index_x)):
                        self.centres[index_x[indx], index_y[indy]] = 1000


class Node():
    def __init__(self, position, parent, gcost, hcost, obstacle):
        self.position = position  # grid coordinates
        self.parent = parent  # pass another node here
        self.gcost = gcost
        self.hcost = hcost

        self.f = self.gcost + self.hcost
        self.heap = self.f + self.hcost  # Used for finding heapmin
        self.obstacle = obstacle  # T or F

    # Used in heap queue (priority queue for min)
    def __lt__(self, neighbour):  # overload operator for heap queue
        # eval heapq based on this value
        # of f + h, which encompasses cases where
        # f = f and we must choose using h value
        return self.heap < neighbour.heap


class A_star():
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.obstacle_list = self.obstacle_list()
        self.start = start
        self.goal = goal
        self.start_grid = self.world2grid(start)
        self.goal_grid = self.world2grid(goal)

        self.open_list = []
        self.closed_list = []
        self.path = []  # this is in world coord
        self.start_node = Node(self.start_grid, None, 0, 0, False)
        self.goal_node = Node(self.goal_grid, None, 0, 0, False)
        self.current_node = Node(
            self.start_grid, None, 0,
            self.get_dist(self.start_node, self.goal_node), False
        )  # init at start  -- NEED TO REPLACE 2ND 0 WITH CALC HEUR FCN

    def obstacle_list(self):
        indxs = np.where(self.grid.centres == 1000)
        coords = list(zip(indxs[0], indxs[1]))

        return coords

    def world2grid(self, coord):
        """ Returns grid coordinates of fed world coordinate values
        """
        for x in range(len(self.grid.cell_xmin)):
            if coord[0] >= self.grid.cell_xmin[x] and coord[
                    0] <= self.grid.cell_xmin[x] + self.grid.cell_size:
                index_x = x
        for y in range(len(self.grid.cell_ymin)):
            if coord[1] >= self.grid.cell_ymin[y] and coord[
                    1] <= self.grid.cell_ymin[y] + self.grid.cell_size:
                index_y = y

        return [index_x, index_y]

    def grid2world(self, coord):
        """ Returns world coordinates of fed grid coordinate values
        """
        c = self.grid.cell_size
        x = c * (coord[0] + (c / 2)) + self.grid.xmin
        y = c * (coord[1] + (c / 2)) + self.grid.ymin
        return [x, y]

    def get_dist(self, node1, node2):
        x_dist = abs(node1.position[0] - node2.position[0])
        y_dist = abs(node1.position[1] - node2.position[1])
        if x_dist > y_dist:
            cost = 1.4 * y_dist + 1.0 * (x_dist - y_dist)
        else:
            cost = 1.4 * x_dist + 1.0 * (y_dist - x_dist)
        # cost = np.sqrt(x_dist**2 + y_dist**2)
        return cost

    def get_neighbours(self, node):
        neighbours = []
        # print("trying to get neighbours")

        # Evaluate about 3x3 block
        for x in range(-1, 2):  # x from -1 to 1
            for y in range(-1, 2):
                # skip at x = 0, y = 0
                if x == 0 and y == 0:
                    continue
                else:
                    check_x = node.position[0] + x
                    check_y = node.position[1] + y
                    # print("Checks: {}".format([check_x, check_y]))

                    # ensure neighbour within grid bounds
                    if check_x >= 0 and check_x < (
                            self.grid.xmax / self.grid.cell_size -
                            self.grid.xmin / self.grid.cell_size
                    ) and check_y >= 0 and check_y < (
                            self.grid.ymax / self.grid.cell_size -
                            self.grid.ymin / self.grid.cell_size):
                        neighbours.append([
                            check_x, check_y
                        ])  # add positions to neighbour lit and compare later

        return neighbours

    def trace_path(self, startnode, currnode):
        self.goal_node = self.current_node

        while self.current_node.position != self.start_node.position:
            self.path.append(self.current_node.position)
            self.current_node = self.current_node.parent

        self.path.reverse()  # reverse path

        # bring back to world coord
        for i in range(len(self.path)):
            self.path[i] = self.grid2world(self.path[i])

        path = self.path
        path.insert(0, self.start)

        return path

    def plan(self):
        self.open_list.append(self.current_node)

        heapq.heapify(self.open_list)

        it = 0
        while len(self.open_list) > 0:
            # print(it)
            # print("\n")
            # print("Open List Size: {}".format(len(self.open_list)))
            # print("The current node is:")
            # pprint(vars(self.current_node))
            it += 1

            self.current_node = self.open_list[0]
            """
            # REPLACED WITH HEAPQ FOR FASTER LOOP
            for i in range(1, len(
                    self.open_list)):  # start at 1 since 0 is current
                # print(" node i cost: {}".format(self.open_list[i].hcost))
                if self.open_list[i].f < self.current_node.f:
                    self.current_node = self.open_list[i]
                elif self.open_list[
                        i].f == self.current_node.f and self.open_list[
                            i].hcost < self.current_node.hcost:
                    self.current_node = self.open_list[i]
            # print("The new current node is:")
            # pprint(vars(self.current_node))

            index_to_pop = self.open_list.index(self.current_node)
            self.open_list.pop(index_to_pop)
            self.closed_list.append(self.current_node)
            """
            # Simultaneously set current node and remove it from openlist
            self.current_node = heapq.heappop(self.open_list)
            # pprint(vars(self.current_node))
            # Heapq unnecessary for closed list
            self.closed_list.append(self.current_node)

            if self.current_node.position == self.goal_node.position:
                return self.trace_path(self.start_node, self.current_node)
                print("goal found after {} iterations!".format(it))
                break

            for neighbour in self.get_neighbours(self.current_node):
                # print("Neighbour pos: {}".format(neighbour))
                skip = False
                closed = False
                opened = False
                obstacle_node = False

                # see if matches coords in closed list
                for node in self.closed_list:
                    if neighbour[0] == node.position[0] and neighbour[
                            1] == node.position[1]:
                        # node exists in closed lit
                        closed = True
                        skip = True

                # see if index matches obstacle list
                for obstacle in self.obstacle_list:
                    if neighbour[0] == obstacle[0] and neighbour[
                            1] == obstacle[1]:
                        obstacle_node = True
                        skip = True

                # see if matches coords in open list
                for node in self.open_list:
                    if neighbour[0] == node.position[0] and neighbour[
                            1] == node.position[1]:
                        # node exists in open list
                        neighbour_node = node
                        opened = True

                # if in none of these lists, create new node
                if closed is False and opened is False and obstacle_node is False:
                    neighbour_temp = Node(neighbour, None, 0, 0, False)
                    h_cost = self.get_dist(neighbour_temp, self.goal_node)
                    g_cost = self.current_node.gcost + self.get_dist(
                        neighbour_temp, self.current_node)
                    neighbour_node = Node(neighbour, self.current_node, g_cost,
                                          h_cost, False)

                    # Push to the right index by comparing .heap
                    # attribute defined in node class under __lt__
                    # (less than)
                    heapq.heappush(self.open_list, neighbour_node)
                    # self.open_list.append(neighbour_node)
                    # print("The chosen node is:")
                    # pprint(vars(neighbour_node))

                elif skip is True:
                    continue
                elif open is True:
                    # h_cost = self.get_dist(neighbour_node, self.goal_node)
                    g_cost = self.current_node.gcost + self.get_dist(
                        neighbour_node, self.current_node)

                    if g_cost < neighbour_node.gcost or self.open_list:
                        neighbour_node.gcost = g_cost
                        neighbour_node.parent = self.current_node
                    # print("The chosen node is:")
                    # pprint(vars(neighbour_node))


"""
class Naive_star():
    def __init__(self, a_grid):
        self.x = 0
"""


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


def plot(landmark_list, a_grid, path):
    """DOCSTRING
    """

    # Initialise Plot
    fig, ax = plt.subplots()

    # plt.imshow adds colour to higher values
    # used to show occupied cells
    plt.imshow(a_grid.centres.T,
               cmap="copper_r",
               origin='lower',
               extent=[a_grid.xmin, a_grid.xmax, a_grid.ymin, a_grid.ymax])
    # extent is left, right, bottom, top

    # Plot Path
    path_x = [x[0] for x in path]
    path_y = [y[1] for y in path]
    plt.plot(path_x, path_y, color='darkviolet', label='A* Path')

    # Plot Path Points
    plt.scatter(path_x,
                path_y,
                color='darkviolet',
                marker='o',
                s=50 * a_grid.cell_size,
                label='Path Points')

    # Plot Start
    plt.plot(path_x[0],
             path_y[0],
             color='g',
             marker='o',
             markersize=10,
             label='Start')

    # Plot Goal
    plt.plot(path_x[-1],
             path_y[-1],
             color='g',
             marker='d',
             markersize=10,
             label='Goal')

    # Set axis ranges
    ax.set_xlim(a_grid.xmin, a_grid.xmax)
    ax.set_ylim(a_grid.ymin, a_grid.ymax)

    # Change major ticks
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Change minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator(0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(0.1))

    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='darkviolet', linestyle='--')
    ax.grid(which='minor', color='darkviolet', linestyle='--')

    plt.title('A* Search')
    plt.axis([-2, 5, -6, 6])
    plt.xlabel('x position [m]')
    plt.ylabel('y position [m]')

    # Plot landmarks
    # Parse F Path
    landmark_x = [x[0] for x in landmark_list]
    landmark_y = [y[1] for y in landmark_list]
    plt.scatter(landmark_x,
                landmark_y,
                alpha=1,
                color='r',
                label='Obstacles',
                s=5)
    # plt.legend()
    plt.show()


def a3(landmark_list, start, goal):
    return True


def a5(landmark_list, start, goal):
    grid_size = 1
    a_grid = Grid(grid_size, landmark_list)
    astar = A_star(a_grid, start, goal)

    path = astar.plan()

    plot(landmark_list, a_grid, path)


def a7(landmark_list, start, goal):
    grid_size = 0.1
    a_grid = Grid(grid_size, landmark_list)
    astar = A_star(a_grid, start, goal)

    path = astar.plan()

    plot(landmark_list, a_grid, path)


# Main
def main():
    # Load Data from ds1 set using Pandas
    landmark_groundtruth = read_dat(
        3, "ds1/ds1_Landmark_Groundtruth.dat",
        ["Subject #", "x [m]", "y [m]", "x std-dev [m]", "y std-dev [m]"])
    landmark_list = []
    for l in range(len(landmark_groundtruth)):
        landmark_list.append(
            [landmark_groundtruth[l][1], landmark_groundtruth[l][2]])

    # Select Exercise
    exercise = raw_input('Select an exercise [3,5,7]')
    if exercise == '3':
        # Exercise 2: Naive Search
        a3(landmark_list, start, goal)
    elif exercise == '5':
        # Exercise 5: A* Search
        input = raw_input('Select a set of coordinates [A, B, C]').upper()
        if input == 'A':
            start = [0.5, -1.5]
            goal = [0.5, 1.5]
        elif input == 'B':
            start = [4.5, 3.5]
            goal = [4.5, -1.5]
        elif input == 'C':
            start = [-0.5, 5.5]
            goal = [1.5, -3.5]

        a5(landmark_list, start, goal)
    elif exercise == '7':
        # Exerise 7: A* with small grid
        input = raw_input('Select a set of coordinates [A, B, C]').upper()
        if input == 'A':
            start = [2.45, -3.55]
            goal = [0.95, -1.55]
        elif input == 'B':
            start = [4.95, -0.05]
            goal = [2.45, 0.25]
        elif input == 'C':
            start = [-0.55, 1.45]
            goal = [1.95, 3.95]
        a7(landmark_list, start, goal)


if __name__ == "__main__":
    main()
