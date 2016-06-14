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

avoidance_type: '-s' or '-b'
	Choose between 'secondary' or 'blocked' avoidance actions, which determine
	how the robot deals with blocked goals. Defaults to none, which results in only
	avoidance decisions by the navigation stack local planner, with no high-level decisions by
	the POMDP policy.

Output
-----------
Goal poses sent to the ROS navigation stack.

"""

__author__ = "Ian Loefgren"
__copyright__ = "Copyright 2016, Cohrint"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ian Loefgren"
__email__ = "ian.loefgren@colorado.edu"
__status__ = "Development"

import rospy
import roslib
import numpy as np
import sys
import math
import logging

from discretePolicyTranslator import discretePolicyTranslator
from geometry_msgs.msg import PoseStamped
from actionlib_msgs.msg import GoalStatusArray
import std_msgs.msg
import tf

from pose import Pose


class goalHandler(object):

	def __init__(self, filename, avoidance_type):
		rospy.init_node('goal_sender')
		self.stuck_buffer = 5
		self.stuck_count = self.stuck_buffer
		self.dpt = discretePolicyTranslator(filename)
		self.pose = Pose('',[0,0,0],'tf',None)
		self.current_status = 3
		self.last_position = self.pose._pose
		self.tf_exist = False
		self.tf_exception_wrapper()
		self.goal_point = self.pose._pose
		self.avoidance = sys.argv[2]
		self.pub = rospy.Publisher('/deckard/move_base_simple/goal',PoseStamped,queue_size=10)
		rospy.sleep(1) #<>TODO: figure out why the hell this works --> solves issue where robot would not move on initialization
		rospy.Subscriber('/deckard/move_base/status',GoalStatusArray,self.callback)
		#print("initial position: " + str(self.pose._pose))

		logging.basicConfig(filename='goalHandler_debug.log',level=logging.DEBUG)

	def tf_exception_wrapper(self):
		"""waits for transforms to become available and handles interim exceptions
		"""
		tries = 0
		while not self.tf_exist and tries < 10:
			try:
				self.pose.tf_update()
				self.tf_exist = True
				#self.current_position = self.pose._pose
				#self.goal_point = self.pose._pose
			except tf.LookupException as error:
				tries = tries + 1
				self.tf_exist = False
				# print("\nError!")
				# print(error)
				# print("Waiting for transforms to become available. Will retry 10 times.")
				# print("Try: " + str(tries) + "  Retrying in 2 seconds.\n")
				error_str = "\nError!\n" + error + "\nWaiting for transforms to become available. Will retry 10 times." \
							+ "\nTry: " + str(tries) + " Retrying in 2 seconds.\n"
				print(error_str)
				logging.error(error_str)
				rospy.sleep(2)

	def callback(self,msg):
		"""callback function that runs when messages are published to /move_base/status.
		The function updates its knowledge of its position using tf data, then
		checks if the robot is stuck and sends the appropriate goal pose.
		"""

		self.pose.tf_update()
		#self.last_position = self.pose._pose
		if self.is_stuck():
			self.send_goal(True)
			while not self.is_at_goal():
				continue
		else:
			self.send_goal()
		#self.is_at_goal()
		#print("status: " + str(self.current_status) + "\tcheck: " + str(self.current_status==3) + "\tcurrent position: " + str(self.pose._pose))
		#if self.current_status == 3:
		#self.stuck_count = self.stuck_buffer
			#self.current_status = 1
		rospy.sleep(1)

	def is_at_goal(self):
		"""checks if robot has arrived at its goal pose
		"""

		tol = 0.25
		#print("Checking if arrived at goal")
		try:
			#print("X goal diff: " + str(abs(self.goal_point[0] - self.current_position[0])) + "\tY goal diff: " + str(abs(self.goal_point[1] - self.current_position[1])))
			if abs(self.goal_point[0] - self.pose._pose[0]) < tol and abs(self.goal_point[1] - self.pose._pose[1]) < tol:
				self.current_status = 3
				return True
		except TypeError:
			print("Goal pose does not yet exist!")
			self.current_status = 3

	def is_stuck(self):
		"""re-sends goal pose if robot is mentally or physically stuck for self.stuck_buffer number of iterations
		"""

		if self.stuck_count > 0: #check buffer
			self.stuck_count += -1
			#print("stuck count "+str(self.stuck_count))
			return False #return not stuck
		self.stuck_count = self.stuck_buffer
		self.stuck_distance = math.sqrt(((self.pose._pose[0] - self.last_position[0]) ** 2) + ((self.pose._pose[1] - self.last_position[1]) ** 2))
		self.last_position = self.pose._pose
		logging.debug('stuck distance: ' + str(self.stuck_distance))
		if self.stuck_distance < 0.5:
			print("Robot stuck; resending goal.")
			logging.info("Robot stuck; resending goal.")
			return True
		else:
			return False

	def get_new_goal(self,current_position,stuck_flag): #<>TODO: refine stuck vs blocked, as the robot can get stuck without technically beiing blocked
		"""get new goal pose from discretePolicyTranslator module
		"""

		point = current_position
		point[0] = round(point[0])
		point[1] = round(point[1])
		next_point = self.dpt.getNextPose
		if stuck_flag and self.avoidance == '-b': #<>TODO: talk to luke about whether or not to keep any of the 'blocked' code
			return self.dpt.getNextPose(point,stuck_flag)
		elif stuck_flag and self.avoidance == '-s':
			return self.dpt.getNextPose(point,False,stuck_flag)
		else:
			return self.dpt.getNextPose(point)

	def send_goal(self,stuck_flag=False):
		"""get and send new goal pose. Returns false without sending pose if pose to send
		is the same as the current pose and the robot is not stuck (meaning it is enroute
		to that pose)
		"""

		new_pose = self.get_new_goal(self.pose._pose,stuck_flag)
		if self.goal_point == new_pose and not stuck_flag:
			return False
		else:
			self.goal_point = new_pose

		new_goal = PoseStamped()
		new_goal.pose.position.x = self.goal_point[0]
		new_goal.pose.position.y = self.goal_point[1]
		new_goal.pose.position.z = self.goal_point[2]
		theta = self.goal_point[3]

		if self.goal_point == [2,2,0,0]: #<>TODO: Fix this gross hack, it makes puppies cry
			theta = 180

		quat = tf.transformations.quaternion_from_euler(0,0,np.deg2rad(theta))
		new_goal.pose.orientation.x = quat[0]
		new_goal.pose.orientation.y = quat[1]
		new_goal.pose.orientation.z = quat[2]
		new_goal.pose.orientation.w = quat[3]
		#<>NOTE: random spinning that occured in gazebo sim does not occur in when run of physical robot

		new_goal.header.stamp = rospy.Time.now()
		new_goal.header.frame_id = 'map'

		self.pub.publish(new_goal)
		logging.debug("sent goal: " + str(self.goal_point))

if __name__ == "__main__":
	gh = goalHandler(sys.argv[1],sys.argv[2])
	rospy.spin()
