#!/usr/bin/env python
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import animation, rc


# Robot class for PF
class Robot():
    def __init__(self, position, num_particles):
        self.position = position  # x, y, theta at world origin
        # Number of particles
        self.num_part = num_particles
        # linear velocity noise
        self.var_v = 0.003
        # angular velocity noise
        self.var_w = 0.003
        # positional heading noise due to linear velocity
        self.noise_1 = 0.01
        # positional heading noise due to angular velocity
        self.noise_2 = 0.01
        # empty numpy array w/N rows, 4 columns for x,y,theta, weight
        self.particles = np.zeros((self.num_part, 4))
        # set initial particle weights (4th col) to be all equal and sum to 1
        self.particles[:, 3] = 1.0 / self.num_part

    def noisy_controls(self, controls):
        """ Add noise to odometry commands

            TUNABLE: var_v, var_w
        """
        ctrl = np.zeros([self.num_part, 2])
        ctrl[:, 0] = controls[1] + np.random.standard_normal(
            self.num_part) * self.var_v
        ctrl[:, 1] = controls[2] + np.random.standard_normal(
            self.num_part) * self.var_w

        return ctrl

    def noisy_pos_update(self, ctrl, dur_comm):
        for i in range(self.num_part):
            # Perform motion update for each particle
            # to get prior belief
            self.particles[i, :3] = np.transpose(
                self.fwd_prop(ctrl[i], self.particles[i, :3], dur_comm))

        return self.particles

    def fwd_prop(self, controls, prev_state, del_t):
        """ Forward Propagate each particle using noisy commands
            and then perturb heading
        """

        v_ctrl = controls[0]
        w_ctrl = controls[1]
        perturb = np.random.normal(self.noise_1 * controls[0]**2 +
                                   self.noise_2 * controls[1]**2)
        if controls[1] != 0:
            # Linear motion update
            # Probabilistic Robotics (pg. 124, fig. 5.3)
            x_new = prev_state[0] - (v_ctrl / w_ctrl) * np.sin(
                prev_state[2]) + (v_ctrl / w_ctrl) * np.sin(prev_state[2] +
                                                            w_ctrl * del_t)
            y_new = prev_state[1] + (v_ctrl / w_ctrl) * np.cos(
                prev_state[2]) - (v_ctrl / w_ctrl) * np.cos(prev_state[2] +
                                                            w_ctrl * del_t)
        if controls[1] == 0:
            # Simultaneous linear and angular motion update
            # Probabilistic Robotics (pg. 124, fig. 5.3)
            x_new = prev_state[0] - (v_ctrl) * np.sin(prev_state[2]) + (
                v_ctrl) * np.sin(prev_state[2] + w_ctrl * del_t)
            y_new = prev_state[1] + (v_ctrl) * np.cos(prev_state[2]) - (
                v_ctrl) * np.cos(prev_state[2] + w_ctrl * del_t)

        # Heading update with perturbation
        theta_new = prev_state[2] + w_ctrl * del_t + perturb * del_t

        update = np.array([[x_new], [y_new], [theta_new]])
        return update

    def pf_mes_update(self, pose_est, map, observed_feature):
        """ Return the measured landmark position as reported by each
            particle. We later compare this to the known gt position
            of each landmark.
        """

        lm_est = np.zeros([self.num_part, 3])

        # Verify that landmark measured, not robot
        if observed_feature[1] in range(6, 21):
            for j in range(self.num_part):
                # Subect number
                curr_feature = observed_feature[1]
                # For each particle
                # x = x_est + range*cos(bearing + heading_est)
                x_lm = pose_est[j, 0] + observed_feature[2] * np.cos(
                    observed_feature[3] + pose_est[j, 2])
                # x = y_est + range*sin(bearing + heading_est)
                y_lm = pose_est[j, 1] + observed_feature[2] * np.sin(
                    observed_feature[3] + pose_est[j, 2])
                # where each particle thinks the landmark is (x,y)
                lm_est[j] = np.array([curr_feature, x_lm, y_lm])

        return lm_est

    def assign_weights(self, mes_est, landmarks_gt, pos_est):
        """ Assign weights to particles based on the accuracy
            of their belief of the measured landmark position
        """
        # get subject number for current measurement (index 0th row because
        # same subject number for all particles - doesn't matter)
        curr_feature = mes_est[0, 0]
        # determine which landmark's x,y position to evaluate based on
        # subject num
        feature_index = np.where(landmarks_gt[:, 0] == curr_feature)
        # Initialise to zero
        normalizer = 0

        for j in range(self.num_part):
            # Sum of all weights for normalization
            normalizer += 1 / np.sqrt((
                (landmarks_gt[feature_index, 1] - mes_est[j, 1])**2) + (
                    (landmarks_gt[feature_index, 2] - mes_est[j, 2])**2))

        for i in range(self.num_part):
            # distance-based weight assignment
            pre_weight = 1 / np.sqrt((
                (landmarks_gt[feature_index, 1] - mes_est[i, 1])**2) + (
                    (landmarks_gt[feature_index, 2] - mes_est[i, 2])**2))
            pos_est[i, 3] = (pre_weight / normalizer)

        return pos_est

    def move(self, control, t_next):
        """
        Note that noise is added to all motion: x,y,thetha
        So it is assumed that during pure rotation, the robot may
        slide, and its x,y position may change as a result
        """
        dt = np.abs(control[0] - t_next)
        x = self.position[0] + control[1] * np.cos(self.position[2]) * dt
        # print(control)
        # print("---")
        y = self.position[1] + control[1] * np.sin(self.position[2]) * dt
        theta = self.position[2] + control[2] * dt
        try:
            x = x[0]
            y = y[0]
            theta = theta[0]
        except IndexError:
            x = x
            y = y
            theta = theta
        self.position = [x, y, theta]
        return np.array([x, y, theta])


def get_times(controls):

    # List of timesteps
    dt_ctrl = np.zeros(np.size(controls[:, 0]))
    # List of time when commands or measurements are given
    ts_ctrl = np.zeros(np.size(controls[:, 0]))
    for i in range(len(controls) - 1):
        del_t = (controls[i + 1][0] - controls[i][0])
        ts_last = controls[i][0]
        dt_ctrl[i] = del_t
        ts_ctrl[i] = ts_last

    return dt_ctrl, ts_ctrl


def barcode_swap(measurements):
    for i in range(len(measurements)):
        if measurements[i, 1] == 5:
            measurements[i, 1] = 1
            continue
        if measurements[i, 1] == 14:
            measurements[i, 1] = 2
            continue
        if measurements[i, 1] == 41:
            measurements[i, 1] = 3
            continue
        if measurements[i, 1] == 32:
            measurements[i, 1] = 4
            continue
        if measurements[i, 1] == 23:
            measurements[i, 1] = 5
            continue
        if measurements[i, 1] == 63:
            measurements[i, 1] = 6
            continue
        if measurements[i, 1] == 25:
            measurements[i, 1] = 7
            continue
        if measurements[i, 1] == 45:
            measurements[i, 1] = 8
            continue
        if measurements[i, 1] == 16:
            measurements[i, 1] = 9
            continue
        if measurements[i, 1] == 61:
            measurements[i, 1] = 10
            continue
        if measurements[i, 1] == 36:
            measurements[i, 1] = 11
            continue
        if measurements[i, 1] == 18:
            measurements[i, 1] = 12
            continue
        if measurements[i, 1] == 9:
            measurements[i, 1] = 13
            continue
        if measurements[i, 1] == 72:
            measurements[i, 1] = 14
            continue
        if measurements[i, 1] == 70:
            measurements[i, 1] = 15
            continue
        if measurements[i, 1] == 81:
            measurements[i, 1] = 16
            continue
        if measurements[i, 1] == 54:
            measurements[i, 1] = 17
            continue
        if measurements[i, 1] == 27:
            measurements[i, 1] = 18
            continue
        if measurements[i, 1] == 7:
            measurements[i, 1] = 19
            continue
        if measurements[i, 1] == 90:
            measurements[i, 1] = 20
            continue
    return measurements


def pf(controls, ground_truth, barcodes, landmarks_gt, sensor_mes, animate):
    robot = Robot(np.array([[0.98038490], [-4.99232180], [1.44849633]]), 1000)

    # Rercord execution time
    start_time = time.time()

    # Subtract starting time to make data more legible
    sensor_mes[:, 0] -= 1288971840
    controls[:, 0] -= 1288971840
    ground_truth[:, 0] -= 1288971840

    sensor_mes = barcode_swap(sensor_mes)

    # # Convert Measurement.dat subject # into Landmark subject # w barcode.dat
    # necessary to ensure no double-replace. ie 16 --> 9 --> 13
    # sensor_mes_check = sensor_mes
    # for b in range(len(barcodes)):
    #     for mes in range(len(sensor_mes)):
    #         if barcodes[b][1] == sensor_mes[mes][1]:
    #             # only replace if value unchanged from original
    #             if sensor_mes[mes][1] == sensor_mes_check[mes][1]:
    #                 sensor_mes[mes][1] = barcodes[b][0]

    # for c in range(len(sensor_mes_1)):
    #     if sensor_mes_1[c][1] != sensor_mes[c][1]:
    #         print("MISMATCH:\n")
    #         print("long: {}".format(sensor_mes_1[c][1]))
    #         print("short: {}".format(sensor_mes[c][1]))

    if (animate == 'y' or animate == 'Y'):
        plt.ion()

    init_pos = np.array([[0.98038490], [-4.99232180], [1.44849633]])

    # Particles: x, y, theta
    robot.particles[:, :3] = robot.position.T

    # Particle Filter Path
    mean_traj = np.array([[0.98038490, -4.99232180, 1.44849633]])

    # Ground Truth Path
    ground_truth_plot = np.array(
        [[1288971842.054 - 1288971840, 0.98038490, -4.99232180, 1.44849633]])

    # Dead Reckoning Path
    dr = np.array([[0.98038490, -4.99232180, 1.44849633]])

    # Get timesteps and issuance times for controls and measurements
    dt_ctrl, ts_ctrl = get_times(controls)
    dt_mes, ts_mes = get_times(sensor_mes)

    # initialise measurement index
    j = 0

    # initialise ground truth counter for plotting
    gt = 0
    # Position model update step
    iterations = 11520
    for i in range(iterations):

        measure_avail = True

        # Only perform PF update if nonzero controls
        if controls[i, 1] == 0.0 and controls[i, 2] == 0.0:
            continue

        if i % 100 == 0:
            print("Loop {} out of {}".format(i, iterations))

        # Generate noisy controls v and w
        ctrl = robot.noisy_controls(controls[i, :])

        # Perform position update using generated noisy controls
        pos_est = robot.noisy_pos_update(ctrl, dt_ctrl[i])

        # Dead Reckoning Update
        t_next = controls[i + 1][0]
        # print(dr)
        # print(robot.move(controls[i], t_next))
        dr = np.vstack((dr, robot.move(controls[i], t_next)))
        # print(dr)

        # measure_avail True for available valid measurements.
        # Apply measurement update
        while measure_avail is True:

            if ts_mes[j] < ts_ctrl[i + 1]:
                # If landmark measured
                if sensor_mes[j, 1] in range(6, 21):
                    # Measurement Update
                    mes_est = robot.pf_mes_update(pos_est, landmarks_gt,
                                                  sensor_mes[j])
                    # Weight update
                    pos_est = robot.assign_weights(mes_est, landmarks_gt,
                                                   pos_est)

                    # Resampling - maybe modify to do lowvar resampling?
                    # Generate random indeces as samples
                    resample_indeces = np.random.choice(robot.num_part,
                                                        robot.num_part,
                                                        replace=True,
                                                        p=pos_est[:, 3])
                    # store resampled particles
                    pos_est = pos_est[resample_indeces[:], :]
                    # Update particles
                    robot.particles = pos_est

                    # Update trajectory
                    curr_mean = np.mean(pos_est[:, :3], axis=0)
                    mean_traj = np.append(mean_traj,
                                          np.array([curr_mean[:3]]),
                                          axis=0)
                # start from this measurement next time
                j += 1
            else:
                # Position update with no measurement update
                curr_mean = np.mean(pos_est[:, :3], axis=0)
                mean_traj = np.append(mean_traj,
                                      np.array([curr_mean[:3]]),
                                      axis=0)
                # Exit loop
                measure_avail = False

        if (animate == 'y'):
            # make sure removed particles are not left as artefacts
            plt.draw()
            for g in range(gt, len(ground_truth)):
                # plot gt elements until latest issued control wrt time
                if ground_truth[g, 0] <= controls[i + 1, 0]:
                    # incr gt counter to start from there next loop
                    gt += 1
                    ground_truth_plot = np.append(ground_truth_plot,
                                                  np.array([ground_truth[g]]),
                                                  axis=0)

            groundtruth_xy, = plt.plot(ground_truth_plot[:, 1],
                                       ground_truth_plot[:, 2],
                                       'g',
                                       label='Groundtruth')
            est_traj, = plt.plot(mean_traj[:, 0],
                                 mean_traj[:, 1],
                                 color='darkviolet',
                                 label='Particle Filter')

            # print(dr[0])

            dr_traj, = plt.plot(dr[:, 0],
                                dr[:, 1],
                                color='red',
                                label='Dead Reckoning')

            initial_state, = plt.plot(init_pos[0],
                                      init_pos[1],
                                      color='gold',
                                      label='Start')
            # lm_groundtruth_plot, = plt.plot(landmarks_gt[:, 1],
            #                                 landmarks_gt[:, 2],
            #                                 'k*',
            #                                 label='Landmarks')
            particles_plot = plt.scatter(pos_est[:, 0],
                                         pos_est[:, 1],
                                         c='orange',
                                         marker='o',
                                         s=1,
                                         label='Current state set')
            plt.title('State Estimation with Particle Filter')
            plt.legend([
                groundtruth_xy,
                est_traj,
                initial_state,
                particles_plot,
                dr_traj,
            ], [
                'Groundtruth', 'Filter', 'Start', 'Particles', 'Dead Reckoning'
            ],
                       loc="upper left")
            plt.ylim(-5.2, -2)
            plt.xlim(0.8, 4.3)
            # Get current plot
            ax = plt.gca()
            # set height/width ratio to 2
            ax.set_aspect(2)
            plt.xlabel('x')
            plt.ylabel('y')
            # plt.show()
            plt.pause(0.00001)
            # remove previously drawn particles
            particles_plot.remove()
    if (animate == 'n'):
        groundtruth_xy, = plt.plot(ground_truth[:, 1],
                                   ground_truth[:, 2],
                                   'g',
                                   label='Groundtruth')
        est_traj, = plt.plot(mean_traj[:, 0],
                             mean_traj[:, 1],
                             color='darkviolet',
                             label='Particle Filter')

        # print(dr[0])

        dr_traj, = plt.plot(dr[:, 0],
                            dr[:, 1],
                            color='red',
                            label='Dead Reckoning')

        initial_state, = plt.plot(init_pos[0],
                                  init_pos[1],
                                  color='gold',
                                  label='Start')
        lm_groundtruth_plot = plt.scatter(landmarks_gt[:, 1],
                                          landmarks_gt[:, 2],
                                          c='black',
                                          marker='o',
                                          label='Landmarks')
        plt.title('Ground Truth VS. Dead Reckoning')
        plt.legend([groundtruth_xy, dr_traj, lm_groundtruth_plot, est_traj],
                   ['Groundtruth', 'Dead Reckoning', 'Landmarks', 'Filter'])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    print('Runtime: %s seconds. \n' % (time.time() - start_time))
    print('Saving mean trajectory to text file mean_traj.txt... \n')
    np.savetxt('mean_traj.txt', mean_traj)
    print('Trajectory saved.')


def main():
    """ The main function """
    animate_choice = raw_input("Animate? [y/n] ")

    # Load data
    controls = np.loadtxt('ds1/ds1_Odometry.dat')
    ground_truth = np.loadtxt('ds1/ds1_Groundtruth.dat')
    barcodes = np.loadtxt('ds1/ds1_Barcodes.dat')
    landmarks_gt = np.loadtxt('ds1/ds1_Landmark_Groundtruth.dat')
    sensor_mes = np.loadtxt('ds1/ds1_Measurement.dat')

    pf(controls, ground_truth, barcodes, landmarks_gt, sensor_mes,
       animate_choice)

    return


if __name__ == '__main__':
    main()