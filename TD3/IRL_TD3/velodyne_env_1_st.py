import math
import os
import random
import subprocess
import time
from os import path
# import tf
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Joy

from sensor_msgs.msg import LaserScan
from numpy import inf
from math import atan2
from math import sqrt
from math import sin
from math import cos
# from tf.transformations import euler_from_quaternion
# from geometry_msgs.msg import Point

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1

# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):

        self.kv = 1
        self.kw = 1
        self.r2_x1 = 0.0
        self.r2_y1 = 0.0
        self.theta1 = 0.0
        self.x_goal = 0.0
        self.goal1 = 0.0
        self.goal2 = 0.0
        self.dg = 5
        self.laser = []
        self.joy_f = 0.0
        self.joy_w = 0.0
        self.joy_en = 0

        # self.position_ = Point()
        # self.yaw_ = 0
        # # machine state
        # self.state_ = 0
        # # goal
        # self.desired_position_ = Point()
        # self.desired_position_.x = -3
        # self.desired_position_.y = 7
        # self.desired_position_.z = 0
        # # parameters
        # self.yaw_precision_ = math.pi / 90  # +/- 2 degree allowed
        # self.dist_precision_ = 0.3
        #
        # # publishers
        # self.pub = None




        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 0.073
        self.goal_y = 1.958
        # self.goal_x = 2.5
        # self.goal_y = 0.171

        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.last_odom = None

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        port = "11311"
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
            # print("launchfile", launchfile)
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
            print("launchfile", launchfile)
            print("fullpath", fullpath)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # setup dynamic obstacle
        # rospy.init_node("controller1")
        self.sub = rospy.Subscriber("r2/front_laser/scan", LaserScan, self.Lidar)
        self.sub1 = rospy.Subscriber('/r2/odom', Odometry, self.new0dom)
        # self.sub3 = rospy.Subscriber('/xy1', Point, self.path1)
        self.pub1 = rospy.Publisher('/r2/cmd_vel', Twist, queue_size=1)
        self.speed = Twist()



        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )
        self.joy = rospy.Subscriber('/joy', Joy, self.newjoy, queue_size=1)
    def newjoy(self, msg):
        self.joy_w = msg.axes[0]
        self.joy_f = msg.axes[1]
        self.joy_en = msg.buttons[0]
    def return_joy(self):
        # print("self.joy_W", self.joy_w)
        # print("self.joy_f", self.joy_f)
        return self.joy_f, self.joy_w

    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def odom_callback(self, od_data):
        self.last_odom = od_data

    # Perform an action and read a new state
    def step(self, action):
        target = False

        # Publish the robot action
        action[0] = self.joy_f
        action[1] = -1*self.joy_w
        # print("joy_en", self.joy_en)
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]

        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate distance between robot1 and robot2
        distance_r1r2 = np.linalg.norm(
            [self.odom_x - self.r2_x1, self.odom_y - self.r2_y1]
        )



        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True
        r1r2_state = False
        if distance_r1r2 <= 0.5:
            r1r2_state = True
        # print("distance_r1r2", distance_r1r2)
        # print("distance", distance)
        # print("min_lser", np.array(min_laser).shape)
        # print("min_laser", min_laser)

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, collision, action, min_laser, r1r2_state)
        return state, reward, done, target, action
    def step_eva(self, action):
        target = False

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate distance between robot1 and robot2
        distance_r1r2 = np.linalg.norm(
            [self.odom_x - self.r2_x1, self.odom_y - self.r2_y1]
        )



        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True
        r1r2_state = False
        if distance_r1r2 <= 0.5:
            r1r2_state = True
        # print("distance_r1r2", distance_r1r2)
        # print("distance", distance)
        # print("min_lser", np.array(min_laser).shape)
        # print("min_laser", min_laser)

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, collision, action, min_laser, r1r2_state)
        return state, reward, done, target, action
    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        # angle = 1.49
        # # angle = 0.12
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        # x = 0.073
        # y = -2.0
        # x = -1.5
        # y = 0.171
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        self.random_box()
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = True

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            # self.goal_x = 0.073
            # self.goal_x = 1.958
            # self.goal_x = 2.5
            # self.goal_x = 0.171
            goal_ok = check_pos(self.goal_x, self.goal_y)


    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "r1/odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "r1/odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "r1/odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        # print("laser_shape", laser_data)
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser, r1r2_state):
        if target:
            return 100.0
        elif collision:
            return -100.0
        elif r1r2_state:
            return -10.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2


    def new0dom(self, msg):
        # global self.x1
        # global self.y1
        global theta1
        self.r2_x1 = msg.pose.pose.position.x
        self.r2_y1 = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        quaternion = Quaternion(msg.pose.pose.orientation.x,
                                msg.pose.pose.orientation.y,
                                msg.pose.pose.orientation.z,
                                msg.pose.pose.orientation.w)
        # (roll, pitch, theta1) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
        (roll, pitch, theta1)= quaternion.to_euler(degrees=False)

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


    def onestep(self, goal1, goal2):
        # while (dg > 1) & (i > 1)
        self.goal1 = goal1
        self.goal2 = goal2
        self.value = np.array(self.laser)
        self.value[self.value == inf] = 20
        self.min_value = min(self.value)
        # print('location(x,y),dg,i', x1, y1, dg, i)
        dx = self.goal1 - self.r2_x1
        dy = self.goal2 - self.r2_y1
        # print("goal_x and goal_y", self.goal1, self.goal2)
        # print("dx and dy", dx, dy)
        # print("dg", self.dg)
        angle_to_goal = atan2(dy, dx)
        e = atan2(sin(angle_to_goal - theta1), cos(angle_to_goal - theta1))
        self.dg = self.kv * sqrt(dx * dx + dy * dy)
        # if 0.1 < abs(e) < 0.3:
        #     self.speed.angular.z = (self.kw * e)
        # self.speed.linear.x = 0.0
        if abs(e) > 0.5:
            emax = 0.3
            emin = -0.3
            ein = max(min(e, emax), emin)
            self.speed.angular.z = (self.kw * ein)
            self.speed.linear.x = 0.0
            # self.pub1.publish(self.speed)
            # print("speed.angular", self.speed.angular.z)
        else:
            v = self.dg
            vmax = 0.8
            v = max(min(v, vmax), -vmax)
            self.speed.linear.x = v
            self.speed.angular.z = 0.0
            # print("v_speed", v)
            # self.pub1.publish(self.speed)
        if self.min_value <= 0.5:
            self.speed.linear.x = 0
            self.speed.angular.z = 0
            self.pub1.publish(self.speed)
        else:
            self.pub1.publish(self.speed)

        # time.sleep(TIME_DELTA)
        return self.dg
# r1 = rospy.Rate(1)

    # callbacks
    # def clbk_odom(self, msg):
    #     # position
    #     self.position_ = msg.pose.pose.position
    #         av_Q += torch.mean(target_Q)
    #         max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
    #
    #     # yaw
    #     quaternion = (
    #         msg.pose.pose.orientation.x,
    #         msg.pose.pose.orientation.y,
    #         msg.pose.pose.orientation.z,
    #         msg.pose.pose.orientation.w)
    #     euler = euler_from_quaternion(quaternion)
    #     self.yaw_ = euler[2]
    #
    # def change_state(self, state):
    #     self.state_ = state
    #     print('State changed to [%s]' % self.state_)
    #
    # def fix_yaw(self, des_pos):
    #     global yaw_, pub, yaw_precision_, state_
    #     desired_yaw = math.atan2(des_pos.y - position_.y, des_pos.x - position_.x)
    #     err_yaw = desired_yaw - yaw_
    #
    #     twist_msg = Twist()
    #     if math.fabs(err_yaw) > yaw_precision_:
    #         twist_msg.angular.z = 0.7 if err_yaw > 0 else -0.7
    #
    #     pub.publish(twist_msg)
    #
    #     # state change conditions
    #     if math.fabs(err_yaw) <= yaw_precision_:
    #         print('Yaw error: [%s]' % err_yaw)
    #         change_state(1)
    #
    # def go_straight_ahead(des_pos):
    #     global yaw_, pub, yaw_precision_, state_
    #     desired_yaw = math.atan2(des_pos.y - position_.y, des_pos.x - position_.x)
    #     err_yaw = desired_yaw - yaw_
    #     err_pos = math.sqrt(pow(des_pos.y - position_.y, 2) + pow(des_pos.x - position_.x, 2))
    #
    #     if err_pos > dist_precision_:
    #         twist_msg = Twist()
    #         twist_msg.linear.x = 0.6
    #         pub.publish(twist_msg)
    #     else:
    #         print('Position error: [%s]' % err_pos)
    #         change_state(2)
    #
    #     # state change conditions
    #     if math.fabs(err_yaw) > yaw_precision_:
    #         print('Yaw error: [%s]' % err_yaw)
    #         change_state(0)
    #
    # def done():
    #     twist_msg = Twist()
    #     twist_msg.linear.x = 0
    #     twist_msg.angular.z = 0
    #     pub.publish(twist_msg)

