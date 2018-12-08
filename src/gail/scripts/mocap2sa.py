#!/usr/bin/env python
import tf
import math
import numpy
import rospy
import roslib

from geometry_msgs.msg import Pose
from std_msgs.msg import Float32

from bvh_broadcaster import BVHBroadcaster

class Mocap2SA:
    '''
    learning end-effector motion
    '''
    def __init__(self):
        self.root_frame = 'world'
        self.tf_listener = tf.TransformListener()
        self.mocap_ee_parent_frame = 'Hips'
        self.mocap_ee_child_frame = 'RightHandIndex1'
        self.ur10_base_frame = 'base_link'
        self.pose_list = []

    def __call__(self):
        self.dt = rospy.get_param('bvh_dt')
        if self.dt == None:
            return
        rate = rospy.Rate(1/self.dt)
        print('Converting')
        while not rospy.is_shutdown():
            rate.sleep()
            self.get_pose()

    def get_pose(self):
        (trans, rot) = self.tf_listener.lookupTransform(self.mocap_ee_child_frame, self.mocap_ee_parent_frame, rospy.Time(0))
        trans_mat = tf.transformations.translation_matrix(trans)
        rot_mat = tf.transformations.quaternion_matrix(rot)
        matrix_b = numpy.matmul(trans_mat, rot_mat)

        (trans, rot) = self.tf_listener.lookupTransform(self.mocap_ee_parent_frame, self.ur10_base_frame, rospy.Time(0))
        trans_mat = tf.transformations.translation_matrix(trans)
        rot_mat = tf.transformations.quaternion_matrix(rot)
        tf_b_s = numpy.matmul(trans_mat, rot_mat)

        matrix_s = numpy.matmul(tf_b_s, matrix_b)

        posi = tf.transformations.translation_from_matrix(matrix_s)
        orien = tf.transformations.quaternion_from_matrix(matrix_s)

        pose_s = Pose(position=posi, orientation=orien)
        self.pose_list.append(pose_s)

    def save_pose(self):
        print('Saving')
        numpy.save('mocap.npy', self.pose_list)
        print('Done')

if __name__ == "__main__":
    rospy.init_node('mocap2sa')
    converter = Mocap2SA()
    try:
        converter()
    except rospy.ROSInterruptException:
        converter.save_pose()