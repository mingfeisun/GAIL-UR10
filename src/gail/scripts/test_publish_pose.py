#!/usr/bin/env python
import tf
import math
import numpy
import rospy
import roslib
import util as U_

from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float32, Header

from bvh_broadcaster import BVHBroadcaster

class PublishPose:
    def __init__(self):
        self.pose_list = []
        self.pose_publisher = rospy.Publisher('mocap_ee_pose', PoseStamped, queue_size=20)

    def __call__(self):
        rate = rospy.Rate(50)
        for po in self.pose_list:
            hdr = Header(stamp=rospy.Time.now(), frame_id='base_link')
            self.pose_publisher.publish(PoseStamped(header=hdr, pose=po))
            rate.sleep()

    def get_pose(self):
        for ob in self.ob_pose:
            pose_s = Pose()
            pose_s.position.x = ob[0] * 2.0
            pose_s.position.y = ob[1] * 2.0
            pose_s.position.z = ob[2] + 0.5
            pose_s.orientation.x = ob[3]
            pose_s.orientation.y = ob[4]
            pose_s.orientation.z = ob[5]
            pose_s.orientation.w = ob[6]
            self.pose_list.append(pose_s)

    def load_pose(self):
        print('Loading...')
        data = numpy.load(U_.getDataPath() + '/obs_acs.npz')
        self.ob_pose = data['obs']
        self.get_pose()
        print('Loading done')

if __name__ == "__main__":
    rospy.init_node('publish_pose')
    publisher = PublishPose()
    publisher.load_pose()
    while not rospy.is_shutdown():
        try:
            publisher()
        except rospy.ROSInterruptException:
            pass