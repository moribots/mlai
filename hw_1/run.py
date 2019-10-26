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
import matplotlib.ticker as plticker
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

        # Evaluate neighbours in +-3 block
        for c in range(len(coords)):
            for x in range(-3, 4):
                for y in range(-3, 4):
                    self.centres[coords[c][0] + x, coords[c][1] + y] = 1000

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
        # heap evaluates first item of tuple first, then second if
        # first item is equal for both nodes
        self.heap = (self.f, self.hcost)  # Used for finding heapmin
        self.obstacle = obstacle  # T or F

    # Used in heap queue (priority queue for min)
    def __lt__(self, neighbour):  # overload operator for heap queue
        # eval heapq based on this value
        return self.heap < neighbour.heap


class A_star():  # G COST IS ALWAYS 1 NO MATTER WHAT
    def __init__(self, grid, start, goal):
        # import grid instance of Grid class
        self.grid = grid
        # set obstacle indeces in grid coords
        self.obstacle_list = self.obstacle_list()
        # Import start coord
        self.start = start
        # Import goal coord
        self.goal = goal
        # Set start and goal coord to grid indeces
        self.start_grid = self.world2grid(start)
        self.goal_grid = self.world2grid(goal)
        # Init open and closed lists, and path
        self.open_list = []
        self.closed_list = []
        self.path = []  # this is in world coord
        self.start_node = Node(self.start_grid, None, 0, 0, False)
        self.goal_node = Node(self.goal_grid, None, 0, 0, False)
        # Init current node with 0 g cost and h cost determined by
        # get_dist method
        self.current_node = Node(
            self.start_grid, None, 0,
            self.get_dist(self.start_node, self.goal_node), False
        )  # init at start  -- NEED TO REPLACE 2ND 0 WITH CALC HEUR FCN

    def obstacle_list(self):
        """ Returns indeces of obstacles in grid coord
        """
        indxs = np.where(self.grid.centres == 1000)
        coords = list(zip(indxs[0], indxs[1]))

        return coords

    def world2grid(self, coord):
        """ Returns grid coordinates of fed world coordinate values
        """
        for x in range(len(self.grid.cell_xmin)):
            if coord[0] >= self.grid.cell_xmin[x] and coord[
                    0] < self.grid.cell_xmin[x] + self.grid.cell_size:
                index_x = x
        for y in range(len(self.grid.cell_ymin)):
            if coord[1] >= self.grid.cell_ymin[y] and coord[
                    1] < self.grid.cell_ymin[y] + self.grid.cell_size:
                index_y = y

        return [index_x, index_y]

    def grid2world(self, coord):
        """ Returns world coordinates of fed grid coordinate values
        """
        c = self.grid.cell_size
        if c == 1:
            x = c * (coord[0] + (c / 2)) + self.grid.xmin
            y = c * (coord[1] + (c / 2)) + self.grid.ymin
        elif c == 0.1:
            x = coord[0] * c - 1.95
            y = coord[1] * c - 5.95
        return [x, y]

    def get_dist(self, node1, node2):
        """Computes heuristic using manhattan distance
        """
        x_dist = abs(node1.position[0] - node2.position[0])
        y_dist = abs(node1.position[1] - node2.position[1])
        D1 = 1
        D2 = 1  # np.sqrt(2)  # or use 1
        cost = D1 * (x_dist + y_dist) + (D2 - 2 * D1) * min(x_dist, y_dist)
        # cost = np.sqrt(x_dist**2 + y_dist**2)
        return cost

    def get_dist_n(self, node1, node2):
        """Computes heuristic using manhattan distance
        """
        x_dist = abs(node1.position[0] - node2.position[0])
        y_dist = abs(node1.position[1] - node2.position[1])
        cost = np.sqrt(x_dist**2 + y_dist**2)
        return cost

    def get_neighbours(self, node):
        """ Evaluates 8 neighbours surrounding each
            node (Cell)
        """
        neighbours = []

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
        """ Traces path from goal to start via
            parent nodes of each node in closed list

            Then converts path coordinates (grid indeces)
            into world coordinates after reversing path
        """
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

    def plan_naive(self):
        """ Main Planning Loop
            for A* (naive)

            Steps:

            1. Append start node to open list
            2. LOOP:
                a- set current node to that with lowest f cost
                in open list. If some nodes have the same f cost,
                choose current node using h cost

                b- get the 8 neighbours of each node (or less depending
                on proximity to grid limits)

                c- for each node:
                    i - if it is inside the closed list,
                    or if it is an obstacle, ignore it
                    ii - if it is on the open list, check
                    whether it is a better path than the current node,
                    if so, update its g cost and make its parent node the
                    current node
                    iii - if none of these conditions are met,
                    add the node to the open list and set its parent
                    to the current node after calculating its g and h cost
            3. If goal is found, return path, else, keep looping
            until goal is found or open list is 0 (path was never found)
        """
        self.open_list.append(self.current_node)

        heapq.heapify(self.open_list)

        it = 0
        while len(self.open_list) > 0:
            it += 1
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
                opened = False

                # see if matches coords in closed list
                for node in self.closed_list:
                    if neighbour[0] == node.position[0] and neighbour[
                            1] == node.position[1]:
                        # node exists in closed lit
                        skip = True

                # see if index matches obstacle list
                for obstacle in self.obstacle_list:
                    if neighbour[0] == obstacle[0] and neighbour[
                            1] == obstacle[1]:
                        skip = True

                # see if matches coords in open list
                for node in self.open_list:
                    if neighbour[0] == node.position[0] and neighbour[
                            1] == node.position[1]:
                        # node exists in open list
                        neighbour_node = node
                        opened = True

                if skip is True:
                    continue

                # if in none of these lists, create new node
                elif opened is False:
                    neighbour_temp = Node(neighbour, None, 0, 0, False)
                    h_cost = self.get_dist(neighbour_temp, self.goal_node)
                    g_cost = self.current_node.gcost + self.get_dist_n(
                        neighbour_temp, self.current_node)
                    neighbour_node = Node(neighbour, self.current_node, g_cost,
                                          h_cost, False)

                    # Push to the right index by comparing .heap
                    # attribute defined in node class under __lt__
                    # (less than)
                    heapq.heappush(self.open_list, neighbour_node)
                    # print("The chosen node is:")
                    # pprint(vars(neighbour_node))
                elif opened is True:
                    g_cost = self.current_node.gcost + self.get_dist(
                        neighbour_node, self.current_node)

                    if g_cost < neighbour_node.gcost:
                        neighbour_node.gcost = g_cost
                        neighbour_node.parent = self.current_node
                    # print("The chosen node is:")
                    # pprint(vars(neighbour_node))

    def plan_online(self):
        """ Main Planning Loop
            for A* (online)

            Steps:

            1. Append start node to open list
            2. LOOP:
                a- set current node to that with lowest f cost
                in open list. If some nodes have the same f cost,
                choose current node using h cost

                b- get the 8 neighbours of each node (or less depending
                on proximity to grid limits)

                c- for each node:
                    i - if it is inside the closed list,
                    or if it is an obstacle, ignore it
                    ii - if none of these conditions are met,
                    add the node to the open list and set its parent
                    to the current node after calculating its g and h cost
                d - only append the node with the lowest f cost to the open
                list. If some nodes have the same f cost, sort using h cost.
            3. If goal is found, return path, else, keep looping
            until goal is found or open list is 0 (path was never found)
        """
        self.open_list.append(self.current_node)

        heapq.heapify(self.open_list)

        it = 0
        while len(self.open_list) > 0:
            # pprint(vars(self.current_node))
            it += 1
            # Simultaneously set current node and remove it from openlist
            self.current_node = heapq.heappop(self.open_list)
            # pprint(vars(self.current_node))
            # Heapq unnecessary for closed list
            self.closed_list.append(self.current_node)

            if self.current_node.position == self.goal_node.position:
                return self.trace_path(self.start_node, self.current_node)
                print("goal found after {} iterations!".format(it))
                break

            self.neighbour_list = []  # restart neighbour list every iteration
            heapq.heapify(self.neighbour_list)

            for neighbour in self.get_neighbours(self.current_node):
                # print("Neighbour pos: {}".format(neighbour))
                skip = False

                # see if matches coords in closed list
                for node in self.closed_list:
                    if neighbour[0] == node.position[0] and neighbour[
                            1] == node.position[1]:
                        # node exists in closed lit
                        skip = True

                # see if index matches obstacle list
                for obstacle in self.obstacle_list:
                    if neighbour[0] == obstacle[0] and neighbour[
                            1] == obstacle[1]:
                        skip = True

                if skip is True:
                    continue
                # if in none of these lists, create new node
                else:
                    neighbour_temp = Node(neighbour, None, 0, 0, False)
                    h_cost = self.get_dist(neighbour_temp, self.goal_node)
                    # g_cost = self.current_node.gcost + self.get_dist(
                    #     neighbour_temp, self.current_node)
                    g_cost = self.current_node.gcost + self.get_dist_n(
                        neighbour_temp, self.current_node)
                    # g_cost = self.current_node.gcost + 1
                    neighbour_node = Node(neighbour, self.current_node, g_cost,
                                          h_cost, False)

                    # Push to the right index by comparing .heap
                    # attribute defined in node class under __lt__
                    # (less than)
                    heapq.heappush(self.neighbour_list, neighbour_node)

            if len(self.neighbour_list) > 0:
                neighbour_node = heapq.heappop(self.neighbour_list)
                # Avoid back-tracking
                self.closed_list.append(neighbour_node)
                heapq.heappush(self.open_list, neighbour_node)

    def plan_live(self, curr):
        """ Main Planning Loop
            for A* (online with real robot motion)

            Steps:

            1. Append start node to open list
            2. LOOP ONCE:
                a- set current node to that with lowest f cost
                in open list. If some nodes have the same f cost,
                choose current node using h cost

                b- get the 8 neighbours of each node (or less depending
                on proximity to grid limits)

                c- for each node:
                    i - if it is inside the closed list,
                    or if it is an obstacle, ignore it
                    ii - if none of these conditions are met,
                    add the node to the open list and set its parent
                    to the current node after calculating its g and h cost
                d - only append the node with the lowest f cost to the open
                list. If some nodes have the same f cost, sort using h cost.
            3. If goal is found, return true
        """
        # Make node position actual robot position
        curr = self.world2grid(curr)
        # print("curr pos: {}".format(curr))
        # print("goal pos: {}".format(self.goal_node.position))
        # Initialised as start for iteration 0
        self.current_node.position = curr

        self.neighbour_list = []  # restart neighbour list every iteration
        heapq.heapify(self.neighbour_list)

        for neighbour in self.get_neighbours(self.current_node):
            # print("Neighbour pos: {}".format(neighbour))
            skip = False

            # see if matches coords in closed list
            for node in self.closed_list:
                if neighbour[0] == node.position[0] and neighbour[
                        1] == node.position[1]:
                    # node exists in closed lit
                    skip = True

            # see if index matches obstacle list
            for obstacle in self.obstacle_list:
                if neighbour[0] == obstacle[0] and neighbour[1] == obstacle[1]:
                    skip = True

            if skip is True:
                continue
            # if in none of these lists, create new node
            else:
                neighbour_temp = Node(neighbour, None, 0, 0, False)
                h_cost = self.get_dist(neighbour_temp, self.goal_node)
                g_cost = self.current_node.gcost + self.get_dist_n(
                    neighbour_temp, self.current_node)
                neighbour_node = Node(neighbour, self.current_node, g_cost,
                                      h_cost, False)

                # Push to the right index by comparing .heap
                # attribute defined in node class under __lt__
                # (less than)
                heapq.heappush(self.neighbour_list, neighbour_node)

        if len(self.neighbour_list) > 0:
            neighbour_node = heapq.heappop(self.neighbour_list)
            # Avoid back-tracking
            self.closed_list.append(neighbour_node)
            end_pos = neighbour_node.position
            start_pos = self.current_node.position
            # overwrite self.current to retain node info
            # for next iteration
            self.current_node = neighbour_node
            start_pos = self.grid2world(start_pos)
            end_pos = self.grid2world(end_pos)
            return end_pos


class Robot():
    def __init__(self, thresh, nodes):
        self.x = nodes[0][0]
        self.y = nodes[0][1]
        self.th = -np.pi / 2  # start heading
        self.max = [0.288, 5.579]
        self.thresh = thresh
        self.noise = []
        self.std = [0.1, 0.1]
        self.dt = 0.1
        self.nodes = nodes
        self.path = []
        self.u_x_prev = 0  # initial
        self.u_y_prev = 0  # initial
        self.u_v_prev = 0  # initial
        self.u_w_prev = 0  # initial
        self.i = 0

    def rk4_intgr(self, x0, u):
        """ Returns position updates for
            given velocity commands after one timestep
            using a Runge-Kutta 4 integrator
        """
        # update calculated here
        k1 = self.dt * self.dynamics(x0, u)
        # print(k1)
        k2 = self.dt * self.dynamics(x0 + k1 / 2, u)
        k3 = self.dt * self.dynamics(x0 + k2 / 2, u)
        k4 = self.dt * self.dynamics(x0 + k3, u)
        # initial plus update
        xnew = x0 + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return xnew

    def dynamics(self, x0, u):
        x_dot = u[0] * np.cos(x0[2]) * (1 + self.noise[0])
        # print(x_dot)
        y_dot = u[0] * np.sin(x0[2]) * (1 + self.noise[0])
        w = u[1] * (1 + self.noise[1])
        return np.array([x_dot, y_dot, w])

    def euler_intgr(self, x0, u):
        xn = x0[0] + u[0] * np.cos(x0[2]) * self.dt
        yn = x0[1] + u[0] * np.sin(x0[2]) * self.dt
        thn = x0[2] + u[1] * self.dt
        x_new = [xn, yn, thn]
        return x_new

    def control(self, goal):
        dist_i = np.sqrt((self.x - goal[0])**2 + (self.y - goal[1])**2)
        dist = dist_i
        Kpv = 0.02
        Kpw = 0.6
        i = 0

        while dist > self.thresh:
            # print(dist - self.thresh)
            # Controller Noise
            self.noise = [
                np.random.normal(0, self.std[0]),
                np.random.normal(0, self.std[1])
            ]
            # print(dist - self.thresh)
            # print(i)
            i += 1
            dist = np.sqrt((self.x - goal[0])**2 + (self.y - goal[1])**2)
            o = goal[1] - self.y
            a = goal[0] - self.x
            bearing = np.arctan2(o, a) - self.th

            # Bias bearing direction if above or below pi
            if bearing >= np.pi:
                bearing -= 2 * np.pi
            if bearing <= -np.pi:
                bearing += 2 * np.pi

            # print(bearing)
            # Set Kpv and Kpw for max v and w
            u = [Kpv * dist, Kpw * bearing]
            u_w = u[1]
            u_v = u[0]
            a_lin = u_v - self.u_v_prev / self.dt
            a_th = u_w - self.u_w_prev / self.dt

            if a_lin > self.max[0]:
                u[0] = self.u_v_prev + self.max[0] * self.dt
                # update linear velocity for next loop
                self.u_v_prev = u[0]
            elif a_lin < -self.max[0]:
                u[0] = self.u_v_prev + self.max[0] * self.dt
                self.u_v_prev = u[0]
            else:
                self.u_v_prev = u[0]

            if a_th > self.max[1]:
                u[1] = self.u_w_prev + self.max[1] * self.dt
                # update angular velocity for next loop
                self.u_w_prev = u[1]
            elif a_th < -self.max[1]:
                u[1] = self.u_w_prev - self.max[1] * self.dt
                self.u_w_prev = u[1]
            else:
                self.u_w_prev = u[1]

            # Issue Control using RK4 intgr
            x0 = np.array([self.x, self.y, self.th])
            new_state = self.rk4_intgr(x0, u)
            # new_state = self.euler_intgr(x0, u)
            # print(new_state)

            # update directional velocities for next loop
            self.u_x_prev = abs(self.x - new_state[0]) / self.dt
            self.u_y_prev = abs(self.y - new_state[1]) / self.dt
            # self.u_w_prev = abs(self.th - new_state[2]) / self.dt

            self.x = new_state[0]
            self.y = new_state[1]
            self.th = new_state[2]
            self.path.append(new_state)

    def move(self):
        for i in range(len(self.nodes) - 1):
            goal = self.nodes[i + 1]
            self.control(goal)
            # print([self.x, self.y, self.th])
            self.i = i
            print("iteration: {}".format(i))
            # print(self.nodes[i][0])
        return self.path

    def move_live(self, curr, goal):
        self.x = curr[0]
        self.y = curr[1]
        print([self.x, self.y, self.th])
        # don't modify self.theta

        self.control(goal)

        return self.path


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


def plot_a(landmark_list, a_grid, path, neighbours, exp_nodes):
    """ Plot path and (naive) expanded nodes
    """
    # Initialise Plot
    fig, ax = plt.subplots()

    # plt.imshow adds colour to higher values
    # used to show occupied cells
    plt.imshow(a_grid.centres.T,
               cmap='Paired',
               origin='lower',
               extent=[a_grid.xmin, a_grid.xmax, a_grid.ymin, a_grid.ymax])
    # extent is left, right, bottom, top

    # Plot Neighbour Nodes
    n_x = [x[0] for x in neighbours]
    n_y = [y[1] for y in neighbours]
    plt.scatter(n_x,
                n_y,
                color='k',
                marker='*',
                s=20 * a_grid.cell_size,
                label='Exp Nodes')

    # Plot Expanded Nodes
    exp_x = [x[0] for x in exp_nodes]
    exp_y = [y[1] for y in exp_nodes]
    plt.scatter(exp_x,
                exp_y,
                color='k',
                marker='*',
                s=20 * a_grid.cell_size,
                label='Exp Nodes')

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

    # Annotate Start
    ax.annotate('Start',
                xy=(path_x[0], path_y[0]),
                xytext=(path_x[0] + 1, path_y[0] - 1),
                arrowprops=dict(facecolor='green', shrink=0.05))
    # Annotate Goal
    ax.annotate('Goal',
                xy=(path_x[-1], path_y[-1]),
                xytext=(path_x[-1] + 1, path_y[-1] + 1),
                arrowprops=dict(facecolor='green', shrink=0.05))

    # Set axis ranges
    ax.set_xlim(a_grid.xmin, a_grid.xmax)
    ax.set_ylim(a_grid.ymin, a_grid.ymax)

    # Change major ticks
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Change minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator(0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(0.1))

    # loc = plticker.MultipleLocator(base=a_grid.cell_size)
    # ax.xaxis.set_major_locator(loc)
    # ax.yaxis.set_major_locator(loc)
    # ax.grid(which='major', axis='both')
    # ax.set_xticks(np.arange(-2, 5, a_grid.cell_size))
    # ax.set_yticks(np.arange(-6, 6, a_grid.cell_size))

    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='darkviolet', linestyle='-')
    ax.grid(which='minor', color='darkviolet', linestyle='-')

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
    grid_size = 1
    a_grid = Grid(grid_size, landmark_list)
    astar = A_star(a_grid, start, goal)

    path = astar.plan_naive()

    open_list = astar.open_list
    open_l = []
    for node in open_list:
        position = astar.grid2world(node.position)
        open_l.append(position)

    closed_list = astar.closed_list
    closed_l = []
    for node in closed_list:
        position = astar.grid2world(node.position)
        closed_l.append(position)

    plot_a(landmark_list, a_grid, path, open_l, closed_l)


def a5(landmark_list, start, goal):
    grid_size = 1
    a_grid = Grid(grid_size, landmark_list)
    astar = A_star(a_grid, start, goal)

    path = astar.plan_online()

    open_list = astar.open_list
    open_l = []
    for node in open_list:
        position = astar.grid2world(node.position)
        open_l.append(position)

    closed_list = astar.closed_list
    closed_l = []
    for node in closed_list:
        position = astar.grid2world(node.position)
        closed_l.append(position)

    plot_a(landmark_list, a_grid, path, open_l, closed_l)


def a7(landmark_list, start, goal, algo):
    grid_size = 0.1
    a_grid = Grid(grid_size, landmark_list)

    astar = A_star(a_grid, start, goal)

    if algo is True:
        path = astar.plan_online()
    elif algo is False:
        path = astar.plan_naive()

    open_list = astar.open_list
    open_l = []
    for node in open_list:
        position = astar.grid2world(node.position)
        open_l.append(position)

    closed_list = astar.closed_list
    closed_l = []
    for node in closed_list:
        position = astar.grid2world(node.position)
        closed_l.append(position)

    plot_a(landmark_list, a_grid, path, open_l, closed_l)


def plot_b(landmark_list, a_grid, path, bot_path):
    """ Plot path and (naive) expanded nodes
    """

    # Initialise Plot
    fig, ax = plt.subplots()

    # plt.imshow adds colour to higher values
    # used to show occupied cells
    plt.imshow(a_grid.centres.T,
               cmap='Paired',
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

    # Plot Robot Path
    bpath_x = [x[0] for x in bot_path]
    bpath_y = [y[1] for y in bot_path]
    plt.plot(bpath_x, bpath_y, color='r', label='A* Path')

    # Plot Robot Path Points
    plt.scatter(bpath_x,
                bpath_y,
                color='r',
                marker='o',
                s=50 * a_grid.cell_size,
                label='Path Points')

    # Plot Heading Arrows
    for i in range(len(bot_path)):
        if i % 100 == 0 or i == 0:
            plt.arrow(bot_path[i][0],
                      bot_path[i][1],
                      0.1 * np.cos(bot_path[i][2]),
                      0.1 * np.sin(bot_path[i][2]),
                      head_width=0.1,
                      color='green',
                      ec='k')

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

    # Annotate Start
    ax.annotate('Start',
                xy=(path_x[0], path_y[0]),
                xytext=(path_x[0] + 1, path_y[0] - 1),
                arrowprops=dict(facecolor='darkviolet', shrink=0.05))
    # Annotate Goal
    ax.annotate('Goal',
                xy=(path_x[-1], path_y[-1]),
                xytext=(path_x[-1] + 1, path_y[-1] + 1),
                arrowprops=dict(facecolor='darkviolet', shrink=0.05))

    # Set axis ranges
    ax.set_xlim(a_grid.xmin, a_grid.xmax)
    ax.set_ylim(a_grid.ymin, a_grid.ymax)

    # Change major ticks
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Change minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator(0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(0.1))

    # loc = plticker.MultipleLocator(base=a_grid.cell_size)
    # ax.xaxis.set_major_locator(loc)
    # ax.yaxis.set_major_locator(loc)
    # ax.grid(which='major', axis='both')
    # ax.set_xticks(np.arange(-2, 5, cell_size))
    # ax.set_yticks(np.arange(-6, 6, cell_size))

    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='darkviolet', linestyle='-')
    ax.grid(which='minor', color='darkviolet', linestyle='-')

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


def a9(landmark_list, start, goal):
    grid_size = 0.1
    a_grid = Grid(grid_size, landmark_list)

    astar = A_star(a_grid, start, goal)
    thresh = 0.005

    path = astar.plan_online()
    robot = Robot(thresh, path)
    bot_path = robot.move()

    plot_b(landmark_list, a_grid, path, bot_path)


def a10(landmark_list, start, goal):
    grid_size = 0.1
    a_grid = Grid(grid_size, landmark_list)

    astar2 = A_star(a_grid, start, goal)
    thresh = 0.005
    done = False

    curr = start
    path = [start]

    robot = Robot(thresh, path)
    i = 0
    while done is False:
        print("iteration: {}".format(i))
        i += 1
        # perform A* plan for current node
        waypoint = astar2.plan_live(curr)
        path.append(waypoint)
        # feed waypoint to robot instance
        # Move robot to next node
        bot_waypoints = robot.move_live(curr, waypoint)
        # Set current node to next node for loop
        curr = [bot_waypoints[-1][0], bot_waypoints[-1][1]]
        check = np.sqrt((curr[0] - goal[0])**2 +
                        (curr[1] - goal[1])**2)
        if check < thresh:
            done is True
            break

    plot_b(landmark_list, a_grid, path, bot_waypoints)


def a11(landmark_list, start, goal, grid_size):
    a_grid = Grid(grid_size, landmark_list)

    astar2 = A_star(a_grid, start, goal)
    thresh = 0.005
    done = False

    curr = start
    path = [start]

    robot = Robot(thresh, path)
    i = 0
    while done is False:
        print("iteration: {}".format(i))
        i += 1
        # perform A* plan for current node
        waypoint = astar2.plan_live(curr)
        path.append(waypoint)
        # feed waypoint to robot instance
        # Move robot to next node
        bot_waypoints = robot.move_live(curr, waypoint)
        # Set current node to next node for loop
        curr = [bot_waypoints[-1][0], bot_waypoints[-1][1]]
        check = np.sqrt((curr[0] - goal[0])**2 +
                        (curr[1] - goal[1])**2)
        if check < thresh:
            done is True
            break

    plot_b(landmark_list, a_grid, path, bot_waypoints)


# Main
def main():
    """ The main function, shows different
        plots for different exercises based on
        user input
    """
    # Load Data from ds1 set using Pandas
    landmark_groundtruth = read_dat(
        3, "ds1/ds1_Landmark_Groundtruth.dat",
        ["Subject #", "x [m]", "y [m]", "x std-dev [m]", "y std-dev [m]"])
    landmark_list = []
    for l in range(len(landmark_groundtruth)):
        landmark_list.append(
            [landmark_groundtruth[l][1], landmark_groundtruth[l][2]])

    # Select Exercise
    exercise = raw_input('Select an exercise [3,5,7,9,10,11]')
    if exercise == '3' or exercise == '5' or exercise == '11':
        input = raw_input('Select a set of coordinates [A, B, C]').upper()
        if input == 'A':
            start = [0.55, -1.55]
            goal = [0.55, 1.55]
        elif input == 'B':
            start = [4.55, 3.55]
            goal = [4.55, -1.55]
        elif input == 'C':
            start = [-0.55, 5.55]
            goal = [1.55, -3.55]
        if exercise == '3':
            # Exercise 3: Naive Search
            a3(landmark_list, start, goal)
        elif exercise == '5':
            # Exercise 5: Online Search
            a5(landmark_list, start, goal)
        elif exercise == '11':
            grid_type = raw_input('Select Coarse or Fine Grid').upper()
            if grid_type == 'COARSE':
                grid_size = 1
            elif grid_type == 'FINE':
                grid_size = 0.1
            a11(landmark_list, start, goal, grid_size)

    elif exercise == '7':
        # Exerise 7: A* with small grid
        input = raw_input('Select a set of coordinates [A, B, C]').upper()
        algo = raw_input('Use Online or Naive Algo?').upper()
        if input == 'A':
            start = [2.45, -3.55]
            goal = [0.95, -1.55]
        elif input == 'B':
            start = [4.95, -0.05]
            goal = [2.45, 0.25]
        elif input == 'C':
            start = [-0.55, 1.45]
            goal = [1.95, 3.95]
        if algo == 'ONLINE':
            algo = True
        elif algo == 'NAIVE':
            algo = False
        a7(landmark_list, start, goal, algo)

    elif exercise == '9' or exercise == '10':
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
        if exercise == '9':
            # Exercise 9, robot motion post-plan
            a9(landmark_list, start, goal)
        elif exercise == '10':
            # Exercise 10, robot motion with plan
            a10(landmark_list, start, goal)


if __name__ == "__main__":
    main()
