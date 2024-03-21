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
from mvp_msgs.srv import GetStateRequest, GetState, ChangeState, ChangeStateRequest
import time

class Wp_Admin:
    def __init__(self):
        rospy.Subscriber("/path", Path, callback=self.path_cB)
        rospy.Subscriber("/alpha_rise/odometry/filtered/local", Odometry, callback=self.odom_cB)
        self.pub_update = rospy.Publisher("/alpha_rise/helm/path_3d/update_waypoints", PolygonStamped, queue_size=1)
        self.pub_append = rospy.Publisher("/alpha_rise/helm/path_3d/append_waypoints", PolygonStamped, queue_size=1)
        self.vx_x, self.vx_y, self.vx_yaw = 0,0,0
        self.start_time = time.time()
        self.state = "survey_3d"

    def odom_cB(self, msg):
        self.vx_x = msg.pose.pose.position.x
        self.vx_y = msg.pose.pose.position.y

        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w

        self.vx_yaw = math.degrees(tf_transform.euler_from_quaternion([x,y,z,w])[2])

    def path_cB(self, msg):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 2:
            self.start_time = time.time()
            service_client_get_state = rospy.ServiceProxy("/alpha_rise/helm/get_state", GetState)
            request = GetStateRequest("")
            response = service_client_get_state(request)
            self.state = response.state.name

        if self.state == "survey_3d":
            point = self.find_point_to_follow(msg)  
            if point != None:

                x_b = point.x
                y_b = point.y

                # print(x_b, y_b)
                wp = PolygonStamped()
                wp.header.stamp = msg.header.stamp
                wp.header.frame_id = msg.header.frame_id
                wp.polygon.points.append(Point32(x_b ,y_b, 0))

                self.pub_update.publish(wp)
                print("switched to survey_3d")

        elif self.state == "start":
            service_client_change_state = rospy.ServiceProxy("/alpha_rise/helm/change_state", ChangeState)
            request = ChangeStateRequest("survey_3d")
            response = service_client_change_state(request)
            print(f"changed to {response.state.name}")

            x_b = self.vx_x + 15
            y_b = self.vx_y

            x_c = x_b 
            y_c = y_b - 15

            wp = PolygonStamped()
            wp.header.stamp = msg.header.stamp
            wp.header.frame_id = msg.header.frame_id
            wp.polygon.points.append(Point32(x_b ,y_b, 0))
            wp.polygon.points.append(Point32(x_c ,y_c, 0))

            self.pub_update.publish(wp)
            print(wp)
            time.sleep(20)

    def find_point_to_follow(self, path_msg):
        farthest_coordinate = None
        max_distance = float('-inf')

        # Iterate through the poses in the path
        for pose_stamped in path_msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y

            distance,direction = self.get_vector([self.vx_x, self.vx_y], [x,y])
            if distance > max_distance:
                if -90 < direction < 90:
                    max_distance = distance

                    farthest_coordinate = pose_stamped.pose.position

        return farthest_coordinate
    
    def get_vector(self, point1, point2):
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