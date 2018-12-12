#!/usr/bin/env python2.7
import rospy
import actionlib

import util as U_

import sys
import copy
import random
import numpy as np
import rospy
import moveit_commander
from moveit_commander import RobotTrajectory
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped
from math import pi, cos, sin

from moveit_commander.conversions import pose_to_list

from control_msgs.msg import *
from trajectory_msgs.msg import *

class RobotControl:
    def __init__(self):
        rospy.init_node('robot_move_ur', anonymous=True)
        self.robot = moveit_commander.RobotCommander()

        self.scene = moveit_commander.PlanningSceneInterface()

        self.group_end = moveit_commander.MoveGroupCommander("endeffector")
        self.group_man = moveit_commander.MoveGroupCommander("manipulator")

        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)

        self.current_pose = self.group_man.get_current_pose().pose

        self.joint_client = actionlib.SimpleActionClient('arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.joint_client.wait_for_server()

        self.orien_x = 0.004
        self.orien_y = 0.710
        self.orien_z = 0.003
        self.orien_w = 0.704

        self.robot_pose = Pose() 
        self.robot_pose.position.x = -0.75 
        self.robot_pose.position.y = 0.0
        self.robot_pose.position.z = 0.8

        # self.addCollision()

    def initRobotPose(self):
        JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        # [144.593816163399, 5.754934304529601, 7.194142435155028, 10.61821127013265, 4.675844406769917, 7.934736338099062]
        Q1 = [0.009653124896662924, -0.6835756311532828, 1.0619281313412259, -0.3737989105267019, 0, 0]
        Q2 = [0.009653124896662924, -0.6835756311532828, 1.170799852990027, -2.05, -1.57, 0]
        # Q3 = [0.009653124896662924, -0.6835756311532828, 1.170799852990027, -1.9876127002995183, 4.681749171284383, 1.8825401280344316]
        # Q2 = [1.5,0,-1.57,0,0,0]
        # Q3 = [1.5,-0.2,-1.57,0,0,0]

        g = FollowJointTrajectoryGoal()

        g.trajectory = JointTrajectory()
        g.trajectory.joint_names = JOINT_NAMES

        g.trajectory.points = [
            JointTrajectoryPoint(positions=Q1, velocities=[0]*6, time_from_start=rospy.Duration(3.0)), 
            JointTrajectoryPoint(positions=Q2, velocities=[0]*6, time_from_start=rospy.Duration(6.0))
            # JointTrajectoryPoint(positions=Q3, velocities=[0]*6, time_from_start=rospy.Duration(6.0))
        ]

        self.joint_client.send_goal(g)
        try:
            self.joint_client.wait_for_result()
        except KeyboardInterrupt:
            self.joint_client.cancel_goal()
    
    '''
    def addCollision(self):
        collision_bottom_pose = geometry_msgs.msg.PoseStamped()
        collision_bottom_pose.header.frame_id = "world"
        collision_bottom_pose.pose.orientation.w = 1.0
        collision_bottom_pose.pose.position.x = 0.4
        collision_bottom_pose.pose.position.z = -0.1
        collision_bottom_name = "collision_bottom"
        self.scene.add_box(collision_bottom_name, collision_bottom_pose, size=(2, 2, 0.2))

        collision_back_pose = geometry_msgs.msg.PoseStamped()
        collision_back_pose.header.frame_id = "world"
        collision_back_pose.pose.orientation.w = 1.0
        collision_back_pose.pose.position.x = -0.5
        collision_back_pose.pose.position.z = 0.4
        collision_back_name = "collision_back"
        self.scene.add_box(collision_back_name, collision_back_pose, size=(0.2, 2, 1))

        collision_top_pose = geometry_msgs.msg.PoseStamped()
        collision_top_pose.header.frame_id = "world"
        collision_top_pose.pose.orientation.w = 1.0
        collision_top_pose.pose.position.x = 0.4
        collision_top_pose.pose.position.z = 1.0
        collision_top_name = "collision_top"
        self.scene.add_box(collision_top_name, collision_top_pose, size=(2, 2, 0.2))
    '''

    def moveArmToPose(self, _pose_goal):
        self.group_man.set_pose_target(_pose_goal)
        self.group_man.go(wait=True)
        self.group_man.stop()
        self.group_man.clear_pose_targets()
        self.current_pose = self.group_man.get_current_pose().pose

    def moveArmTo(self, _x, _y, _z):
        pose_goal = self.generateRobotPose(_x, _y, _z)
        (plan, _) = self.group_man.compute_cartesian_path([pose_goal], 0.01, 0.0)
        self.group_man.execute(plan, wait=True)
        self.current_pose = self.group_man.get_current_pose().pose

    def generateRobotPose(self, _x, _y, _z):
        pose_goal = geometry_msgs.msg.Pose()

        pose_goal.orientation.x = self.orien_x
        pose_goal.orientation.y = self.orien_y
        pose_goal.orientation.z = self.orien_z
        pose_goal.orientation.w = self.orien_w

        pose_goal.position.x = _x
        pose_goal.position.y = _y
        pose_goal.position.z = _z

        return pose_goal
    
    def reset(self):
        # need to return end-effector pose
        self.initRobotPose()
        return self.group_man.get_current_pose().pose

    def _act(self, positions, velocities):
        JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        g = FollowJointTrajectoryGoal()
        g.trajectory = JointTrajectory()
        g.trajectory.joint_names = JOINT_NAMES
        g.trajectory.points = [ JointTrajectoryPoint(positions=positions, velocities=velocities, time_from_start=rospy.Duration(0.5))]

        self.joint_client.send_goal(g)
        try:
            self.joint_client.wait_for_result()
        except KeyboardInterrupt:
            self.joint_client.cancel_goal()

        return self.group_man.get_current_pose().pose

    def step(self, ac):
        return self._act(ac[:6], ac[6:])

    def traj_generator(self, pi, env, horizon, stochastic, reward_giver):
        t = 0
        ob = self.reset()
        cur_ep_ret = 0  # return in current episode
        cur_ep_len = 0  # len of current episode

        # Initialize history arrays
        obs = []
        rews = []
        acs = []

        while True:
            ac, vpred = pi.act(stochastic, ob)
            rew = reward_giver.get_reward(ob, ac)
            obs.append(ob)
            acs.append(ac)
            rews.append(rew)
            ob = self.step(ac)
            if t >= horizon:
                break
            t += 1
        obs = np.array(obs)
        acs = np.array(acs)
        traj = {"ob": obs, "ac": acs, "rew": rews}
        return traj

    def obs2actions(self, _obs_list):
        assert isinstance(_obs_list[0], Pose)

        n_obs = len(_obs_list)

        waypoints = []
        for obs in _obs_list:
            waypoints.append(copy.deepcopy(obs))

        self.moveArmToPose(waypoints[0])

        (plan, _) = self.group_man.compute_cartesian_path(waypoints[1:], 0.001, 0.0)
        n_points = len(plan.joint_trajectory.points)

        self.group_man.execute(plan, wait=True)

        step_length = n_points//n_obs

        x_state = []
        y_action = []

        for idx in range(n_obs):
            tmp_state = _obs_list[idx]
            x_state.append([tmp_state.position.x, 
                            tmp_state.position.y, 
                            tmp_state.position.z, 
                            tmp_state.orientation.x,
                            tmp_state.orientation.y,
                            tmp_state.orientation.z,
                            tmp_state.orientation.w
                            ])
            tmp_action = plan.joint_trajectory.points[idx]
            tmp_action_list = []
            tmp_action_list.extend(tmp_action.positions)
            tmp_action_list.extend(tmp_action.velocities)
            # tmp_action_list.extend(tmp_action.accelerations)
            y_action.append(tmp_action_list)

        x_state = np.array(x_state)
        y_action = np.array(y_action)

        rospy.loginfo('Obs-acs pairs saved!')
        np.savez(U_.getDataPath() + '/obs_acs.npz', obs=x_state, acs=y_action)

if __name__ == "__main__":
    test = RobotControl()
    # test.initRobotPose()

    obs_list = np.load(U_.getDataPath() + '/mocap.npy')
    test.obs2actions(obs_list)

    # while not rospy.is_shutdown():
    #     waypoints = []
    #     waypoints.append(test.generateRobotPose(0.7, -0.7, 0.1))
    #     waypoints.append(test.generateRobotPose(0.4, -0.4, 0.4))
    #     waypoints.append(test.generateRobotPose(0.7, -0.7, 0.09))
    #     (plan, _) = test.group_man.compute_cartesian_path(waypoints, 0.01, 0.0)
    #     test.group_man.execute(plan, wait=True)
