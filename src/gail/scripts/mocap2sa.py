#!/usr/bin/env python
import tf
import math
import numpy
import rospy
import roslib

import geometry_msgs.msg

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

    def __call__(self):
        frame_matrix = []
        motion_matrix = []

        for ur10_frame, mocap_frame in zip(self.ur10_frames_parent, self.mocap_frames_parent):
            try:
                (trans, rot) = self.tf_listener.lookupTransform(mocap_frame, ur10_frame, rospy.Time(0))
                trans_mat = tf.transformations.translation_matrix(trans)
                rot_mat = tf.transformations.quaternion_matrix(rot)
                frame_matrix.append(numpy.matmul(trans_mat, rot_mat))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass

        for child, parent in zip(self.mocap_frames_child, self.mocap_frames_parent):
            try:
                (trans, rot) = self.tf_listener.lookupTransform(mocap_frame, ur10_frame, rospy.Time(0))
                trans_mat = tf.transformations.translation_matrix(trans)
                rot_mat = tf.transformations.quaternion_matrix(rot)
                motion_matrix.append(numpy.matmul(trans_mat, rot_mat))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass

        for motion, frame_tf in zip(motion_matrix, frame_matrix):
            self.converted_motion.append(numpy.matmul(frame_matrix, motion_matrix))

        print(self.converted_motion)

if __name__ == "__main__":
    rospy.init_node('mocap2sa')
    converter = Mocap2SA()
    rospy.sleep(0.5)
    converter()