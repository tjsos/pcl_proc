#!/usr/bin/env python3

#Author: Tony Jacob
#Part of RISE Project. 
#Assigns waypoints from the path to be fed to the vx nav system.
#tony.jacob@uri.edu

import rospy
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float32
from geometry_msgs.msg import PolygonStamped, Point32
import tf.transformations as tf_transform
import math
from mvp_msgs.srv import GetStateRequest, GetState, ChangeState, ChangeStateRequest
import time

class Wp_Admin:
    def __init__(self):
        #Get Params
        self.distance_in_meters = rospy.get_param("path_generator/standoff_distance_meters",15)
        path_topic = rospy.get_param("path_generator/path_topic", "/path")
        update_waypoint_topic = rospy.get_param("waypoint_admin/update_waypoint_topic", "/alpha_rise/helm/path_3d/update_waypoints")
        append_waypoint_topic = rospy.get_param("waypoint_admin/append_waypoint_topic", "/alpha_rise/helm/path_3d/append_waypoints")
        self.get_state_service = rospy.get_param("waypoint_admin/get_state_service", "/alpha_rise/helm/get_state")
        self.change_state_service = rospy.get_param("waypoint_admin/change_state_service", "/alpha_rise/helm/change_state")
        self.slope_tolerance = rospy.get_param("path_generator/slope_tolerance", 1)
        #Declare Pubs
        self.pub_update = rospy.Publisher(update_waypoint_topic, PolygonStamped, queue_size=1)
        self.pub_append = rospy.Publisher(append_waypoint_topic, PolygonStamped, queue_size=1)
        
        #Declare Subs
        rospy.Subscriber(path_topic, Path, callback=self.path_cB)
        rospy.Subscriber(path_topic+"/slope", Float32, callback=self.slope_cB)
        rospy.Subscriber("/alpha_rise/odometry/filtered/local", Odometry, callback=self.odom_cB)

        #Declare variables
        self.vx_x, self.vx_y, self.vx_yaw, self.current_slope = 0,0,0,0
        self.start_time = time.time()
        self.state = "survey_3d"
    
    def slope_cB(self, msg):
        self.current_slope = msg.data

    def odom_cB(self, msg):
        self.vx_x = msg.pose.pose.position.x
        self.vx_y = msg.pose.pose.position.y

        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w

        self.vx_yaw = math.degrees(tf_transform.euler_from_quaternion([x,y,z,w])[2])

    def path_cB(self, msg):
        self.check_state()
        #Create Waypoint Message
        wp = PolygonStamped()
        wp.header.stamp = msg.header.stamp
        wp.header.frame_id = msg.header.frame_id
        if self.state == "survey_3d":
            global_point = self.find_point_to_follow(msg)  
            print(global_point)
            if global_point != None:
                x_c = global_point.x
                y_c = global_point.y
                
                wp.polygon.points.append(Point32(x_c ,y_c, 0))

                self.pub_update.publish(wp)

        elif self.state == "start":
            #Switch state to survey_3d
            service_client_change_state = rospy.ServiceProxy(self.change_state_service, ChangeState)
            request = ChangeStateRequest("survey_3d")
            response = service_client_change_state(request)
            print(f"changed to {response.state.name}")
            
            #Extend the track behaviour
            if self.current_slope >= self.slope_tolerance:
                print("behaviour_1", time.time())
                x_b, y_b = self.extend_line_from_point([self.vx_x, self.vx_y], self.vx_yaw, length=self.distance_in_meters)
                wp.polygon.points.append(Point32(x_b ,y_b, 0))

            #Extend and turn behaviour 
            elif self.current_slope < self.slope_tolerance:
                print("behaviour_2",time.time())
                x_b, y_b = self.extend_line_from_point([self.vx_x, self.vx_y], self.vx_yaw, length=self.distance_in_meters)
                x_c, y_c = self.extend_line_from_point([x_b, y_b], self.vx_yaw-45, length=self.distance_in_meters)
                wp.polygon.points.append(Point32(x_b ,y_b, 0))
                wp.polygon.points.append(Point32(x_c ,y_c, 0))

            self.pub_update.publish(wp)

    def check_state(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 2:
            self.start_time = time.time()
            service_client_get_state = rospy.ServiceProxy(self.get_state_service, GetState)
            request = GetStateRequest("")
            response = service_client_get_state(request)
            self.state = response.state.name

    def extend_line_from_point(self, point, orientation, length):
        # Convert orientation from degrees to radians
        angle_radians = math.radians(orientation)

        # Calculate the coordinates of the second point
        x2 = point[0] + length * math.cos(angle_radians)
        y2 = point[1] + length * math.sin(angle_radians)

        return (x2, y2)
    
    def find_point_to_follow(self, path_msg):
        farthest_coordinate = None
        max_distance = float('-inf')

        # Iterate through the poses in the path
        for pose_stamped in path_msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y

            distance,direction = self.get_vector([self.vx_x, self.vx_y], [x,y])
            if distance > max_distance:
                if abs(self.vx_yaw - direction) < 60:
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