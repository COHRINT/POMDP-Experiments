# POMDP-Experiments
<<<<<<< HEAD
Using Partially Observable Markov Decision Processes to augment movement and question planning
=======
Using Partially Observable Markov Decision Processes to augment movement and question planning.
>>>>>>> 25ab547539a52f6ee8dda51996e6e35344529020

## Overview

### Installation and Run Instructions

For instructions on installing and setting up experiment see the [installation wiki page](https://github.com/COHRINT/POMDP-Experiments/wiki/Installation-Instructions), and for instructions on running the experiment, see the [run instructions page.](https://github.com/COHRINT/POMDP-Experiments/wiki/Run-Instructions)

<<<<<<< HEAD
This experiment uses a Turtlebot with the iRobot Create base and ROS Indigo as its base hardware and software respectively, as well as a VICON camera system to track the postion of the robot. Tutorials to setup Turtlebots, download and install ROS Indigo, and setup VICON can be found using the links below.

- insert links here for tutorials
- insert links here for tutorials
- insert links here for tutorials
=======
This experiment uses a Turtlebot with the iRobot Create base and ROS Indigo as its hardware and software respectively. Also used is a VICON Tracker camera system to track the postion of the robot. Tutorials to setup Turtlebots, download and install ROS Indigo, and setup VICON can be found using the links below.

- [ROS Indigo installation](http://wiki.ros.org/indigo/Installation)
- [VICON Tracker tutorials](http://www.vicon.com/video/#tog-438)
- [Turtlebot information and tutorials](http://wiki.ros.org/Robots/TurtleBot) and [more information](http://learn.turtlebot.com/)
>>>>>>> 25ab547539a52f6ee8dda51996e6e35344529020

### Package Requirements

In addition to all ROS Indigo Turtlebot software, the following ROS packages are required to run the experiment:
- vicon_bridge
<<<<<<< HEAD

### Known Issues
Please see the [issues page in the github respository](http://github.com/COHRINT/POMDP-Experiments/issues) for more detailed descriptions of known issues and bugs and to report any additional bugs you find.

- Delay of one second after making goal_handler node a publisher and before being made a subscriber necessary to ensure robot responds to first goal pose sent.
- Robot oscillates in place when given a goal pose behind itself, with a 180 degree orientation difference.
- When spiralling inward, the last pose has been hardcoded to make the robot turn around to 180 degrees before stopping in order to enable outward spirallin directly afterward. (Related to above bug)
=======
- rosbridge_suite

### Known Issues
_Please see the [issues page in the github respository](http://github.com/COHRINT/POMDP-Experiments/issues) for more detailed descriptions of known issues and bugs and to report any additional bugs you find._

- Robot oscillates in place when given a goal pose behind itself, with a 180 degree orientation difference, most likely because of the cost function used by the Navigation Stack. It begins turning, then, because of that turning, turning the other direction costs less, etc.
- The stuck buffer and secondary goal features are currently not working because goal poses are updated so frequenty that once a secondary goal pose is sent, it is overridden by the original pose. The original pose blocking that allowed secondary goals to work has been removed because it was clunky and interferred more than it helped for the Tag Avoid problem.
>>>>>>> 25ab547539a52f6ee8dda51996e6e35344529020
