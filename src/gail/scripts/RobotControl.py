#!/usr/bin/env python
import rospy
import actionlib

import sys
import copy
import random
import numpy
import rospy
import moveit_commander
from moveit_commander import RobotTrajectory
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import Pose
from math import pi, cos, sin

from moveit_commander.conversions import pose_to_list

from control_msgs.msg import *
from trajectory_msgs.msg import *

class RobotControl:
    def __init__(self):
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

        self.addCollision()

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
    
if __name__ == "__main__":
    rospy.init_node('robot_move_ur', anonymous=True)
    test = RobotControl()
    test.initRobotPose()

    while not rospy.is_shutdown():
        waypoints = []
        waypoints.append(test.generateRobotPose(0.7, -0.7, 0.1))
        waypoints.append(test.generateRobotPose(0.4, -0.4, 0.4))
        waypoints.append(test.generateRobotPose(0.7, -0.7, 0.09))
        (plan, _) = test.group_man.compute_cartesian_path(waypoints, 0.01, 0.0)
        test.group_man.execute(plan, wait=True)
