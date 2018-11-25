## GAIL-UR10
[Robot manipulation course project]
As a recent breakthrough, [**Generative Adversarial Imitation Learning**](https://arxiv.org/pdf/1606.03476.pdf) (GAIL) provides a new possibility for robots to generate human-like movement patterns from demonstrations. In general, GAIL compares state-action pairs from demonstrations against state-action pairs from the policy by a discriminate classifier (discriminator). The imitation policy is assigned rewards for fooling the discriminator. One of the key advantages of GAIL over previously RLfD techniques, i.e., behavior cloning and IRL, is that the similarity between imitation and demonstration is defined implicitly. Studies have demonstrated that the GAIL produces reusable motions for humanoid robots from only a small number of noisy human motion capture demonstrations. 

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
* Baseline: transformation from BVH to UR10 robot arm
* GAIL method:
* * [Source codes](https://github.com/mingfeisun/GAIL-janathan)
* * Modification
* Training and testing GAIL
* Slides to show demo (10min, due date: 22rd Nov)

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
