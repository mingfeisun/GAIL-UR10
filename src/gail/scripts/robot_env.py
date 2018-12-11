import gym
import numpy as np
import robot_control as RC

class RobotEnv(object):
    def __init__(self):
        self.robot = RC.RobotControl()
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(1, ))
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(1, ))

    def reset(self):
        curr_pose = self.robot.reset()
        return np.array(self.pose2vec(curr_pose))

    def step(self, _ac):
        curr_pose = self.robot.step(_ac)
        return np.array(self.pose2vec(curr_pose))

    def pose2vec(self, _pose):
        return [_pose.position.x, 
                _pose.position.y,
                _pose.position.z,
                _pose.orientation.x,
                _pose.orientation.y,
                _pose.orientation.z,
                _pose.orientation.w ]

    def set_ob_shape(self, _ob_shape):
        self.ob_shape = _ob_shape
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(_ob_shape, ))

    def set_ac_shape(self, _ac_shape):
        self.ac_shape = _ac_shape
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(_ac_shape, ))

    def get_ob_shape(self):
        return self.ob_shape

    def get_ac_shape(self):
        return self.ac_shape
