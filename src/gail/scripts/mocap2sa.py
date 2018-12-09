#!/usr/bin/env python
import tf
import math
import numpy
import rospy
import roslib

from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float32, Header

from bvh_broadcaster import BVHBroadcaster

class Mocap2SA:
    '''
    learning end-effector motion
    '''
    def __init__(self):
        self.root_frame = 'world'
        self.tf_listener = tf.TransformListener()
        self.mocap_ee_parent_frame = 'Hips'
        # self.mocap_ee_parent_frame = 'LeftLeg'
        self.mocap_ee_child_frame = 'RightHandIndex1'
        # self.mocap_ee_child_frame = 'RightHand'
        self.ur10_base_frame = 'base_link'
        self.pose_list = []

        self.pose_publisher = rospy.Publisher('mocap_ee_pose', PoseStamped, queue_size=20)

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
        coord_adjust_matrix = tf.transformations.rotation_matrix(math.pi, [0, 0, 1])

        (trans, rot) = self.tf_listener.lookupTransform(self.mocap_ee_parent_frame, self.mocap_ee_child_frame, rospy.Time(0))
        trans_mat = tf.transformations.translation_matrix(trans)
        rot_mat = tf.transformations.quaternion_matrix(rot)

        matrix_b = numpy.matmul(trans_mat, rot_mat)

        matrix_b = numpy.matmul(matrix_b, coord_adjust_matrix)

        root_adjust_matrix = numpy.array([ [0., 0., 1., 0.], 
                                           [1., 0., 0., 0.],
                                           [0., 1., 0., 0.], 
                                           [0., 0., 0., 1.]])

        matrix_s = numpy.matmul(root_adjust_matrix, matrix_b)

        posi = tf.transformations.translation_from_matrix(matrix_s)
        orien = tf.transformations.quaternion_from_matrix(matrix_s)

        pose_s = Pose()
        pose_s.position.x = posi[0] * 2.0
        pose_s.position.y = posi[1] * 2.0
        pose_s.position.z = posi[2] + 0.5
        pose_s.orientation.x = orien[0]
        pose_s.orientation.y = orien[1]
        pose_s.orientation.z = orien[2]
        pose_s.orientation.w = orien[3]

        hdr = Header(stamp=rospy.Time.now(), frame_id='base_link')
        self.pose_publisher.publish(PoseStamped(header=hdr, pose=pose_s))

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
        pass