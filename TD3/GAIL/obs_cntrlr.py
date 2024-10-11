#! /usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int16
from math import atan2
from math import sqrt
from math import sin
from math import cos
import sys
import numpy as np
from numpy import inf
kv = 1
kw = 1
x1 = 0.0
y1 = 0.0
theta1 = 0.0
x_goal = 0.0
goal1 = 0.0
goal2 = 0.0
dg=5
laser = []

class controller:
    def __init__(self):
        self.kv = 1
        self.kw = 1
        self.x1 = 0.0
        self.y1 = 0.0
        self.theta1 = 0.0
        self.x_goal = 0.0
        self.goal1 = 0.0
        self.goal2 = 0.0
        self.dg = 5
        self.laser = []
        rospy.init_node("controller1")
        self.sub = rospy.Subscriber("r2/front_laser/scan", LaserScan, self.Lidar)
        self.sub1 = rospy.Subscriber('/r2/odom', Odometry, self.new0dom)
        # self.sub3 = rospy.Subscriber('/xy1', Point, self.path1)
        self.pub1 = rospy.Publisher('/r2/cmd_vel', Twist, queue_size=1)
        # self.pub2 = rospy.Publisher('/Ack1', Int16, queue_size=1)

        self.speed = Twist()
        i = 0
        # self.r = rospy.Rate(1)
        # self.r1 = rospy.Rate(1000)

        self.dg = 5  # any value greater then .01

    def new0dom(self, msg):
        # global self.x1
        # global self.y1
        global theta1
        self.x1 = msg.pose.pose.position.x
        self.y1 = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        (roll, pitch, theta1) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

    def path1(self, msg):
        # global goal1
        # global goal2
        self.goal1 = -3.32
        self.goal2 = 0.171

    def Lidar(self, msg):
        # global laser
        self.laser = msg.ranges

    def emergencystop(self):
        self.value = np.array(self.laser)
        self.value[self.value == inf] = 20
        self.min_value = min(self.value)
        if self.min_value <= 0.5:
            v = 0
            w = 0
            return (v, w)
        return


    def onestep(self):
        # while (dg > 1) & (i > 1)
        self.value = np.array(self.laser)
        self.value[self.value == inf] = 20
        self.min_value = min(self.value)
        # print('location(x,y),dg,i', x1, y1, dg, i)
        dx = self.goal1 - self.x1
        dy = self.goal2 - self.y1
        print("goal_x and goal_y", self.goal1, self.goal2)
        # print("dx and dy", dx, dy)
        print("dg", self.dg)
        angle_to_goal = atan2(dy, dx)
        e = atan2(sin(angle_to_goal - theta1), cos(angle_to_goal - theta1))
        self.dg = kv * sqrt(dx * dx + dy * dy)
        if 0.1 < abs(e) < 0.3:
            self.speed.angular.z = (kw * e)
        self.speed.linear.x = 0.0
        if abs(e) > 0.1:
            self.speed.angular.z = (kw * e)
            self.speed.linear.x = 0.0
        else:
            v = dg
            vmax = 0.8
            v = max(min(v, vmax), -vmax)
            self.speed.linear.x = v
            self.speed.angular.z = 0.0
        if self.min_value <= 1:
            self.speed.linear.x = 0
            self.speed.angular.z = 0
            self.pub1.publish(self.speed)
        else:
            self.pub1.publish(self.speed)

        # self.r1.sleep()

        #
        #     # if stop(laser, 0.5):
        #     #   pub1.publish(speed)
        #     # else:
        #     # speed.linear.x = 0
        #     # speed.angular.z = 0
        # if i == 1:
        #     pub2.publish(1)
        #     print('visited=', i)
        # if dg < 1:
        #     pub2.publish(1)  # sending acknowledgment  to path for next subgoal
        #     print('visited=', i)
        #
        # i = i + 1
        # r.sleep()
        # speed.linear.x = 0
        # speed.angular.z = 0
        # pub1.publish(speed)


