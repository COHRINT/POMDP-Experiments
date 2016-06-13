# POMDP-Experiments
Using Partially Observable Markov Decision Processes to augment movement and question planning

## Installation and Run Instructions
This experiment uses a Turtlebot with the iRobot Create base and ROS Indigo as its base hardware and software respectively, as well as a VICON camera system to track the postion of the robot. Tutorials to setup Turtlebots, download and install ROS Indigo, and setup VICON can be found using the links below.

- insert links here for tutorials
- insert links here for tutorials
- insert links here for tutorials

## Package Requirements

In addition to all ROS Indigo Turtlebot software, the following ROS packages are required to run the experiment:
- vicon_bridge

## Known Issues
Please see the issues page in the github respository, http://github.com/COHRINT/POMDP-Experiments/issues for more detailed descriptions of known issues and bugs and to report any additional bugs you find.

- Delay of one second after making goal_handler node a publisher and before being made a subscriber necessary to ensure robot responds to first goal pose sent.
- Robot oscillates in place when given a goal pose behind itself, with a 180 degree orientation difference.
- When spiralling inward, the last pose has been hardcoded to make the robot turn around to 180 degrees before stopping in order to enable outward spirallin directly afterward. (Related to above bug)
