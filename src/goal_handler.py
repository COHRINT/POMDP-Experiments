#!/usr/bin/env python

"""
Written by Ian Loefgren
May 2016

The goalHandler class publishes goal poses to the navigation stack via /move_base_simple/goal.

An instance of goalHandler initilizes a ROS node that published to *robot_name*/move_base_simple/goal, subscribes to
*robot_name*/move_base/status for callbacks, and listens to tf transforms using an instance of the imported "pose" class.

The goal poses come from discretePolicyTranslator.py, which takes the current xy position and returns the desired
xy position and orientation based on a POMDP policy.

Input
-----------
filename: .txt file
	Text file containing alpha vectors for use in discretePolicyTranslator
	(contained in the repository in the "policies" folder)

robot_name:
	String of the robot's name that this script will be generating goal poses for.

other_name:
	String of the name of the other robot in the experiment.

robo_type: '-c' or '-r'
	Flag to specify if the robot is to be the cop or the robber in the tag scenario.
Output
-----------
Goal poses sent to the ROS navigation stack.


Version
----------
1.0.0:
	One robot follows a policy to spiral inward or outward. Secondary avoidance
	(robot picks second best goal) also implemented. No multi-robot capabilities.
1.1.0:
	Implemented two-robot capability in order to run a simple two-robot tag avoid
	problem. Robot names are used to create namespaces, and tf lookups are done
	for all robots' positions.

"""

__author__ = "Ian Loefgren"
__copyright__ = "Copyright 2016, Cohrint"
__license__ = "GPL"
__version__ = "1.1.0"
__maintainer__ = "Ian Loefgren"
__email__ = "ian.loefgren@colorado.edu"
__status__ = "Development"

import rospy
import roslib
import numpy as np
import sys
import math
import logging
import random

#from discretePolicyTranslator import discretePolicyTranslator
#from tagAvoidPolicyTranslator import tagAvoidPolicyTranslator
from continuousPolicyTranslator import continuousPolicyTranslator
import CPerseus
from CPerseus import GM
from CPerseus import Gaussian
from CPerseus import Perseus
from geometry_msgs.msg import PoseStamped
from actionlib_msgs.msg import GoalStatusArray
import std_msgs.msg
import tf

from pose import Pose


class GoalHandler(object):

	def __init__(self, filename, robot_name, other_robot, bot_type):

		#logger = logging.getLogger(__name__)
		logger_level = logging.INFO
		#logger.setLevel(logger_level)
		logger_format = '[%(levelname)-7s] %(funcName)-30s %(message)s'
		try:
			logging.getLogger().setLevel(logger_level)
			logging.getLogger().handlers[0]\
                .setFormatter(logging.Formatter(logger_format))
		except IndexError:
			logging.basicConfig(format=logger_format,
                                level=logger_level,
                               )

		# <>TODO: get this file logging working, as well as debug level
		logger = logging.getLogger('pomdp_log_handle')
		log_filename = robot_name + '_goalHandler_log.log'
		file_handler = logging.FileHandler(log_filename)
		formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
		file_handler.setLevel(logging.DEBUG)

		node_name = robot_name + '_goal_sender'
		pub_name = '/' + robot_name + '/move_base_simple/goal'
		sub_name = '/' + robot_name + '/move_base/status'

		rospy.init_node(node_name,log_level=rospy.DEBUG)

		# Link node to Python's logger
		handler = logging.StreamHandler()
		handler.setFormatter(logging.Formatter(logger_format))
		logging.getLogger().addHandler(handler)

		self.secondary_behavior = False
		self.stuck_buffer = 5
		self.stuck_count = self.stuck_buffer
		self.current_status = 3 #<>NOTE: DEPRECEATED
		self.robo_type = bot_type
		self.robot = robot_name

		self.pt = continuousPolicyTranslator(filename,hardware=True)

		self.pose = Pose(robot_name,[0,0,0],'tf',None)
		if other_robot == "False":
			self.multi = False
		else:
			self.multi = True
			self.other_pose = Pose(other_robot,[2,2,0],'tf',None) #<>TODO: make sure [1,1,1] (or any starting coords) doesn't actually affect anything

		self.last_position = self.pose._pose
		if self.multi:
			self.other_robo_position = self.other_pose._pose

		self.tf_exist = False
		self.tf_exception_wrapper()
		self.goal_point = self.pose._pose

		self.pub = rospy.Publisher(pub_name,PoseStamped,queue_size=10)
		rospy.sleep(1) #<>TODO: figure out why the hell this works --> solves issue where robot would not move on initialization
		rospy.Subscriber(sub_name,GoalStatusArray,self.callback)

		logging.info("Running experiment...")

	def tf_exception_wrapper(self):
		"""waits for transforms to become available and handles interim exceptions
		"""
		tries = 0
		while not self.tf_exist and tries < 10:
			try:
				self.pose.tf_update()
				if self.multi:
					self.other_pose.tf_update()
				self.tf_exist = True
			except tf.LookupException as error:
				tries = tries + 1
				self.tf_exist = False
				error_str = "\nError!\n" + str(error) + "\nWaiting for transforms to become available. Will retry 10 times." \
							+ "\nTry: " + str(tries) + " Retrying in 2 seconds.\n"
				print(error_str)
				logging.error(error_str)
				rospy.sleep(2)

	def callback(self,msg):
		"""callback function that runs when messages are published to /move_base/status.
		The function updates its knowledge of its position using tf data, then
		checks if the robot is stuck and sends the appropriate goal pose.
		"""
		logging.info('called back')
		self.pose.tf_update()
		if self.multi:
			self.other_pose.tf_update()
		logging.info(self.robot + '\'s position: ' + str(self.pose._pose))
		if self.is_stuck():
			self.send_goal(True)
			# while not self.is_at_goal(): #<>NOTE: these lines will re-enable the old obstacle avoidance abiltiy until better method is developed
			# 	self.pose.tf_update()
			# 	rospy.sleep(0.1) #<>TODO: see if anything can be done to avoid adding these sleeps
			# 	logging.info('waiting to reach goal; looping')
		elif self.is_at_goal():
		#else:
			self.send_goal()

		#if (self.robo_type == "-c") and (self.tapt.distance(self.pose._pose[0],self.pose._pose[1],self.other_pose._pose[0],self.other_pose._pose[1]) < 1):
		#	raise KeyboardInterrupt("THE ROBBER WAS CAUGHT!")

		rospy.sleep(1) #<>TODO: test shorter delays as well as no delay at all.

	def is_at_goal(self): #<>NOTE: DEPRECEATED
		"""checks if robot has arrived at its goal pose
		"""
		tol = 0.2
		try:
			if abs(self.goal_point[0] - self.pose._pose[0]) < tol and abs(self.goal_point[1] - self.pose._pose[1]) < tol:
				self.current_status = 3
				return True
		except TypeError:
			print("Goal pose does not yet exist!")
			self.current_status = 3

		return False

	def is_stuck(self):
		"""re-sends goal pose if robot is mentally or physically stuck for self.stuck_buffer number of iterations
		"""
		if self.stuck_count > 0: #check buffer
			self.stuck_count += -1
			#logging.info("stuck count "+str(self.stuck_count))
			return False #return not stuck

		self.stuck_count = self.stuck_buffer
		self.stuck_distance = math.sqrt(((self.pose._pose[0] - self.last_position[0]) ** 2)\
		 		+ ((self.pose._pose[1] - self.last_position[1]) ** 2))

		self.last_position = self.pose._pose
		#logging.info('stuck distance: ' + str(self.stuck_distance))
		if self.stuck_distance < 0.5 and self.current_status != 'final goal':
			print("Robot stuck; resending goal.")
			logging.info("Robot stuck; resending goal.")
			return True
		else:
			return False

	def rotation_assist(self):
		"""sends goal with current position but better orientation to enable
		robot to make 180 degree turns.
		"""
		goal_point = [self.pose._pose[0],
						self.pose._pose[1],0.0,
						self.pose._pose[2]+90]

		new_goal = self.create_goal_msg(goal_point)
		self.pub.publish(new_goal)
		logging.info("sent goal: " + str(self.goal_point))
		rospy.sleep(2)

	def get_new_goal(self,current_position,other_position,stuck_flag): #<>TODO: refine stuck vs blocked, as the robot can get stuck without technically beiing blocked
		"""get new goal pose from policy translator module for multi-robot case
		"""
		logging.info('other robot\'s position: ' + str(other_position))

		if self.robo_type == '-c':
			pose = [current_position[0],current_position[1],other_position[0],other_position[1]]
			#if stuck_flag:
			#	return self.pt.getNextPose(pose,stuck_flag)
			#else:
			return self.pt.getNextPose(pose,True)
		elif self.robo_type == '-r':
			pose = [other_position[0],other_position[1],current_position[0],current_position[1]]
			#if stuck_flag:
			#	return self.pt.getNextPose(other_position,current_position)
			#else:
			return self.pt.getNextPose(pose,False)

	def get_new_goal_singular(self,current_position,stuck_flag):
		"""get new goal pose from policy translator module for single robot case
		"""
		if stuck_flag:
			return self.pt.getNextPose(current_position)
		else:
			return self.pt.getNextPose(current_position)

	def create_goal_msg(self,goal_point):
		"""create message to publish to ROS
		"""
		new_goal = PoseStamped()
		new_goal.pose.position.x = self.goal_point[0]
		new_goal.pose.position.y = self.goal_point[1]
		new_goal.pose.position.z = self.goal_point[2]
		theta = self.goal_point[3]

		quat = tf.transformations.quaternion_from_euler(0,0,np.deg2rad(theta))
		new_goal.pose.orientation.x = quat[0]
		new_goal.pose.orientation.y = quat[1]
		new_goal.pose.orientation.z = quat[2]
		new_goal.pose.orientation.w = quat[3]

		new_goal.header.stamp = rospy.Time.now()
		new_goal.header.frame_id = 'map'

		return new_goal

	def send_goal(self,stuck_flag=False):
		"""get and send new goal pose. Returns false without sending pose if pose to send
		is the same as the current pose and the robot is not stuck (meaning it is enroute
		to that pose)
		"""
		if self.multi:
			new_pose = self.get_new_goal(self.pose._pose,self.other_pose._pose,stuck_flag)
		else:
			new_pose = self.get_new_goal(self.pose._pose,stuck_flag)

		self.goal_point = new_pose #<>TODO: hack to make robot be able to turn 180 degrees deleted, need to make new, better method

		if (abs(new_pose[3] - self.pose._pose[2])) > 120:
			self.rotation_assist()

		new_goal = self.create_goal_msg(new_pose)

		self.pub.publish(new_goal)
		self.current_status = 1
		logging.info("sent goal: " + str(self.goal_point))

if __name__ == "__main__":
	gh = GoalHandler(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
	rospy.spin()
