#!/usr/bin/env python
# coding=utf-8

import rospy
import signal
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry

class CrossDemo:
    def __init__(self, robot_name):
        signal.signal(signal.SIGINT, self.sigint_handler)
        self.robot_name = robot_name
        self.cmd_twist = Twist()
        self.pose = Pose()
        self.cmd_pub = rospy.Publisher(self.robot_name+'/cmd_vel', Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.robot_name+'/odom', Odometry, self.odom_callback)
        self.Main()
        rospy.spin()
    
    def Main(self):
        while self.pose.position.y < 3:
            # print(self.pose.position.y)
            self.CarMove(1, 0)
            rospy.sleep(0.5)
        self.CarMove(0, 0)
        rospy.loginfo("Racecar reached, stop!")

    def odom_callback(self, msg):
        self.pose = msg.pose.pose

    def CarMove(self, x, z):
        self.cmd_twist.linear.x = x
        self.cmd_twist.angular.z = z
        self.cmd_pub.publish(self.cmd_twist)

    def sigint_handler(self, signum, frame):
        self.CarMove(0, 0)
        rospy.logwarn("Catched interrupt signal! Stop and exit...")
        exit()

if __name__ == '__main__':
    rospy.init_node("cross_demo_node")
    robot_name = rospy.get_param('~robot_name', 'AKM_1')
    try:
        CrossDemo(robot_name)
    except Exception as e:
        rospy.logerr(e)
    