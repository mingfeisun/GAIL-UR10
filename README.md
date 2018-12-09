## GAIL-UR10
[Robot manipulation course project]
As a recent breakthrough, [**Generative Adversarial Imitation Learning**](https://arxiv.org/pdf/1606.03476.pdf) (GAIL) provides a new possibility for robots to generate human-like movement patterns from demonstrations. In general, GAIL compares state-action pairs from demonstrations against state-action pairs from the policy by a discriminate classifier (discriminator). The imitation policy is assigned rewards for fooling the discriminator. One of the key advantages of GAIL over previously RLfD techniques, i.e., behavior cloning and IRL, is that the similarity between imitation and demonstration is defined implicitly. Studies have demonstrated that the GAIL produces reusable motions for humanoid robots from only a small number of noisy human motion capture demonstrations. 

A demo to showcase MoCap GAIL

<img src="docs/walking.gif" alt="sawing_robot" width="550px"/>

## TODO list
### Mocap Data processing
* Source mocap data [Subject #62(construction work, random motions)](http://mocap.cs.cmu.edu/search.php?subjectnumber=62)
* Two demos

*62_03*

<img src="mocap/62_03_video.gif" alt="demo1" height="200px"/> <img src="mocap/62_03_ani.gif" alt="demo1" height="200px"/>

*62_07*

<img src="mocap/62_07_video.gif" alt="demo2" height="200px"/> <img src="mocap/62_07_ani.gif" alt="demo2" height="200px"/>

* Broadcasting BVH as tf to rviz, e.g., the guy is laughing

<img src="docs/tf_broadcaster.gif" alt="tf_broadcasting" width="550px"/>

### UR10 robot interface
* DOF control interface

<img src="docs/ur10_frames.png" alt="ur10" width="550px"/>

* Robot motions (sawing-like)

<img src="docs/sawing_robot.gif" alt="sawing_robot" width="550px"/>

### GAIL codes
* [To install dependencies](https://github.com/mingfeisun/GAIL-janathan)
* To run code

* ~~Mocap data: [subject #62(construction work, random motions)](http://mocap.cs.cmu.edu/search.php?subjectnumber=62)~~
* ~~UR10 robot interface: DoF control~~
* ~~Baseline: transformation from BVH to UR10 robot arm~~
* GAIL method:
* * ~~[Source codes](https://github.com/mingfeisun/RL_baselines/tree/master/baselines/gail)~~

<img src="docs/gail_result.gif" alt="sawing_robot" width="550px"/>

* * Modification
* MoCap GAIL: [source codes](https://github.com/ywchao/merel-mocap-gail)
* Training and testing GAIL
* ~~Slides to show demo (10min, due date: 22rd Nov)~~

### TODO
* To define the state and actions for UR10 robot arm: what's the learned policy?

*state* definition (observations)

[hopper.py](https://github.com/openai/gym/blob/master/gym/envs/mujoco/hopper.py#L24)
``` python
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])
```

*action* definition

[mujoco_env.py](https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py#L98)
``` python
    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()
```

[RobotMoveURRobot](https://github.com/mingfeisun/TransparencyRLfD/blob/master/src/main/scripts/RobotMoveURRobot.py#L130)
``` python
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
```

* To find the state and actions for MoCap data: what's the expert policy?

[collect_cmu_mocap.py](https://github.com/ywchao/baselines/blob/05a64203efbfac6eba630001d56b48e950d4d885/data/collect_cmu_mocap.py#L19)
``` python
def main(args):
    env = humanoid_CMU.run()

    print('collecting cmu mocap data ... ')

    assert args.data_path is not None
    amc_qpos = []
    amc_obs = []
    for i in _SUBJECT_AMC_ID:
        file_path = os.path.join(args.data_path,
                                 'subjects','{:02d}'.format(_SUBJECT_ID),
                                 '{:02d}_{:02d}.amc'.format(_SUBJECT_ID,i))
        converted = parse_amc.convert(
            file_path, env.physics, env.control_timestep())

        qpos_seq = []
        obs_seq = []
        for t in range(converted.qpos.shape[-1] - 1):
            p_t = converted.qpos[:, t + 1]
            v_t = converted.qvel[:, t]
            qpos_seq.append(np.concatenate((p_t, v_t))[None])
            with env.physics.reset_context():
                env.physics.data.qpos[:] = p_t
                env.physics.data.qvel[:] = v_t
            obs = get_humanoid_cmu_obs(env)
            obs_seq.append(obs[None])

        amc_qpos.append(np.concatenate(qpos_seq))
        amc_obs.append(np.concatenate(obs_seq))

    if not os.path.isfile(args.save_file):
        np.savez(args.save_file, obs=amc_obs, qpos=amc_qpos)

    print('done.')
```
[dm_control_util.py](https://github.com/ywchao/baselines/blob/05a64203efbfac6eba630001d56b48e950d4d885/baselines/common/dm_control_util.py#L7)
``` python
def get_humanoid_cmu_obs(env):
    # Assumes env is an instance of `HumanoidCMU`.
    obs = env.task.get_observation(env.physics)
    # Add head to extremities
    torso_frame = env.physics.named.data.xmat['thorax'].reshape(3, 3)
    torso_pos = env.physics.named.data.xpos['thorax']
    torso_to_head = env.physics.named.data.xpos['head'] - torso_pos
    positions = torso_to_head.dot(torso_frame)
    obs['extremities'] = np.hstack((obs['extremities'], positions))
    # Remove joint_angles, head_height, velocity
    del obs['joint_angles']
    del obs['head_height']
    del obs['velocity']
    # Flatten observation
    return flatten_observation(obs)['observations']
```

* Learning the end-effector motions:
*Generator*:
* * observations: end-effector poses + twist
* * action: joint configurations + twist
* * policy: actions --> observations
*Discriminator*:
* * GAN: generated observations <-- --> demonstrated observations

* To convert MoCap state+action space to UR10 robot arm
* To construct GAN: *G(s, a)* and *D(s, a)*

## Building & Running
### Dependencies
* [Universal Robot](https://github.com/mingfeisun/universal_robot)
* [Gazebo 8.0](https://github.com/mingfeisun/gazebo)
* [Gazebo ros pkg](https://github.com/mingfeisun/gazebo_ros_pkgs)

### Building
``` bash
# compiling ros pkgs
catkin_make
```
