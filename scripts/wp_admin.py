#!/usr/bin/env python3

#Author: Tony Jacob
#Part of RISE Project. 
#Generates curve, standoff curve of the iceberg from a costmap
#tony.jacob@uri.edu

import rospy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PolygonStamped, Point32
import tf.transformations as tf_transform
import math

class Wp_Admin:
    def __init__(self):
        rospy.Subscriber("/path", Path, callback=self.path_cB)
        rospy.Subscriber("/alpha_rise/odometry/filtered/local", Odometry, callback=self.odom_cB)
        self.pub = rospy.Publisher("/alpha_rise/helm/path_3d/update_waypoints", PolygonStamped, queue_size=1)
        self.vx_x, self.vx_y, self.vx_yaw = 0,0,0

    def odom_cB(self, msg):
        self.vx_x = msg.pose.pose.position.x
        self.vx_y = msg.pose.pose.position.y

        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w

        self.vx_yaw = math.degrees(tf_transform.euler_from_quaternion([x,y,z,w])[2])

    def path_cB(self, msg):
        point = self.find_point_with_highest_x(msg)  
        if point != None:  
            x_b = point.x
            y_b = point.y

            # print(x_b, y_b)
            wp = PolygonStamped()
            wp.header.stamp = msg.header.stamp
            wp.header.frame_id = msg.header.frame_id
            wp.polygon.points.append(Point32(x_b ,y_b, 0))

            self.pub.publish(wp)
            print(f"sent, {x_b}, {y_b}")

    def find_point_with_highest_x(self, path_msg):
        farthest_coordinate = None
        max_distance = float('-inf')

        # Iterate through the poses in the path
        for pose_stamped in path_msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y

            distance,direction = self.distance_between_points([self.vx_x, self.vx_y], [x,y])
            if distance > max_distance:
                if -10 < direction < 90:
                    max_distance = distance

                    farthest_coordinate = pose_stamped.pose.position

        return farthest_coordinate
    
    def distance_between_points(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.
        """
        distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        # Get angle between vehicle and wp
        theta_radians = math.atan2(delta_y, delta_x)
        direction = math.degrees(theta_radians)
        return distance, direction

if __name__ == "__main__":
    rospy.init_node("waypoint_administrator")
    Wp_Admin()
    rospy.spin()