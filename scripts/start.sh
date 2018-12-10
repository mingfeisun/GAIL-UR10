#!/bin/bash
cd ..
catkin_make
source devel/setup.bash
roslaunch gail demo.launch
