#!/usr/bin/env python3

#Author: Tony Jacob
#Part of RISE Project. 
#Assigns waypoints from the path to be fed to the vx nav system.
#tony.jacob@uri.edu

#rosbag record /alpha_rise/path/current_state /alpha_rise/path/distance_to_obstacle /alpha_rise/odometry/filtered/local
import rospy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PolygonStamped, Point32, PointStamped, PoseStamped
from std_msgs.msg import Float32
import tf.transformations as tf_transform
import math
from mvp_msgs.srv import GetStateRequest, GetState, ChangeState, ChangeStateRequest
import time
import tf2_ros
import tf2_geometry_msgs
import numpy as np

class Wp_Admin:
    def __init__(self):
        #Get Params
        self.distance_in_meters = rospy.get_param("path_generator/standoff_distance_meters",15)
        path_topic = rospy.get_param("path_generator/path_topic")
        update_waypoint_topic = rospy.get_param("waypoint_admin/update_waypoint_topic")
        append_waypoint_topic = rospy.get_param("waypoint_admin/append_waypoint_topic")
        state_topic = rospy.get_param("waypoint_admin/state_topic")
        distance_to_obstacle_topic = rospy.get_param("path_generator/distance_to_obstacle_topic")
        odom_topic = rospy.get_param("waypoint_admin/odom_topic")
        self.get_state_service = rospy.get_param("waypoint_admin/get_state_service", "/alpha_rise/helm/get_state")
        self.change_state_service = rospy.get_param("waypoint_admin/change_state_service", "/alpha_rise/helm/change_state")

        #Waypoint selection param
        self.update_rate = rospy.get_param("waypoint_admin/check_state_update_rate")
        self.min_scan_angle = rospy.get_param("waypoint_admin/min_scan_angle")
        self.max_scan_angle = rospy.get_param("waypoint_admin/max_scan_angle")
        self.acceptance_threshold = rospy.get_param("waypoint_admin/acceptance_threshold")
        
        #Declare Pubs
        self.pub_update = rospy.Publisher(update_waypoint_topic, PolygonStamped, queue_size=1)
        self.pub_append = rospy.Publisher(append_waypoint_topic, PolygonStamped, queue_size=1)
        self.pub_state = rospy.Publisher(state_topic, Float32, queue_size=1)
        
        #Declare Subs
        rospy.Subscriber(path_topic, Path, callback=self.path_cB)
        rospy.Subscriber(distance_to_obstacle_topic, Float32, callback= self.distance_cB)
        rospy.Subscriber(odom_topic, Odometry, callback=self.odom_cB)

        #Declare variables
        self.vx_yaw = 0
        self.start_time = time.time()
        self.state = None
        self.distance_to_obstacle = None

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.node_name = rospy.get_name()
    
    def distance_cB(self, msg):
        self.distance_to_obstacle = msg.data

    def odom_cB(self, msg):
        self.lin_x = msg.twist.twist.linear.x
        self.ang_z = msg.twist.twist.angular.z

    def path_cB(self, msg):
        #Get ODOM to VX TF
        self.odom_to_base_tf = self.tf_buffer.lookup_transform("alpha_rise/base_link", "alpha_rise/odom", 
                                                               rospy.Time(0), rospy.Duration(1.0))
        
        #Vx position and bearing in Odom frame.
        self.base_to_odom_tf = self.tf_buffer.lookup_transform("alpha_rise/odom", "alpha_rise/base_link", 
                                                               rospy.Time(0), rospy.Duration(1.0))
        self.vx_x = self.base_to_odom_tf.transform.translation.x
        self.vx_y = self.base_to_odom_tf.transform.translation.y
        self.vx_yaw = tf_transform.euler_from_quaternion([self.base_to_odom_tf.transform.rotation.x,
                                            self.base_to_odom_tf.transform.rotation.y,
                                            self.base_to_odom_tf.transform.rotation.z,
                                            self.base_to_odom_tf.transform.rotation.w])[2]
        self.vx_yaw = math.degrees(self.vx_yaw)
        
        #Create Waypoint Message
        wp = PolygonStamped()
        wp.header.stamp = msg.header.stamp
        wp.header.frame_id = msg.header.frame_id
        
        #Check state whether survey_3d or start
        self.check_state()
        goal = self.find_point_to_follow(msg)  
        if self.state == "survey_3d":
            if goal != None:
                x_c = goal.pose.position.x
                y_c = goal.pose.position.y
                
                wp.polygon.points.append(Point32(x_c ,y_c, 0))

                self.pub_update.publish(wp)
                self.pub_state.publish(0)

        elif self.state == "start":
            #Switch state to survey_3d
            service_client_change_state = rospy.ServiceProxy(self.change_state_service, ChangeState)
            request = ChangeStateRequest("survey_3d", self.node_name)
            response = service_client_change_state(request)
            corner_bhvr_points = self.corner_behaviour(number_of_points=20)
        
            #Append the waypoints
            for i in range(len(corner_bhvr_points)):
                wp.polygon.points.append(Point32(corner_bhvr_points[i].point.x ,corner_bhvr_points[i].point.y, 0))
                
            self.pub_update.publish(wp)

    def check_state(self):
        elapsed_time = time.time() - self.start_time
        #Check state every 2s
        if elapsed_time > self.update_rate:
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
        #Init lists
        valid_pose = []
        valid_xy = []
        valid_polar = []
        relative_yaw = []

        next_point_in_vx_frame = PoseStamped()
        # Iterate through the poses in the path
        for index, pose_stamped in enumerate(path_msg.poses):
            #Transform the points in vx frame
            point_in_vx_frame = tf2_geometry_msgs.do_transform_pose(path_msg.poses[index], self.odom_to_base_tf)
            #Take the next points. If it is last point.
            if index > 20:
                next_point_in_vx_frame = tf2_geometry_msgs.do_transform_pose(path_msg.poses[index+1], self.odom_to_base_tf)
            else:
                next_point_in_vx_frame.pose.position.x = point_in_vx_frame.pose.position.x
                next_point_in_vx_frame.pose.position.y = point_in_vx_frame.pose.position.y

            #Get distance and bearing to the point from the vehicle
            distance_vx,direction_vx,direction_rel = self.get_vector([0,0], [point_in_vx_frame.pose.position.x,
                                                         point_in_vx_frame.pose.position.y],
                                                         
                                                         [next_point_in_vx_frame.pose.position.x,
                                                          next_point_in_vx_frame.pose.position.y])
            #Filter out a sector. REP 103 convention. +ve is counterclockwise from vehicle nose.
            if self.min_scan_angle < direction_vx and direction_vx < self.max_scan_angle:
                #Gather pose_msgs, xy, and distances of such points
                valid_pose.append(pose_stamped)
                valid_xy.append([point_in_vx_frame.pose.position.x,
                                point_in_vx_frame.pose.position.y])
                valid_polar.append(distance_vx)
                relative_yaw.append([direction_vx, direction_rel])
        
        valid_polar_norm_velocity = [distance_vx/self.lin_x for distance_vx in valid_polar]
        relative_yaw_norm_velocity = [(np.radians(direction_vx)/self.ang_z) + (np.radians(direction_rel)/self.ang_z) for direction_vx, direction_rel in relative_yaw]

        time_for_point = [linx_time + angz_time for linx_time, angz_time in zip(valid_polar_norm_velocity, relative_yaw_norm_velocity)]
        try:
            min_value = min(time_for_point)
            min_index = time_for_point.index(min_value)
            all_y_less_than_threshold = all(y < self.acceptance_threshold for x,y in valid_xy)
            
            if not all_y_less_than_threshold:
                print("BEST POINT BHVR", time.time())
                print(min_value)
                return valid_pose[min_index]
            
            else:
                farthest_point = None
                max_distance = float('-inf')
                for index, point in enumerate(valid_pose):
                    if valid_polar[index] > max_distance:
                        max_distance = valid_polar[index]
                        farthest_point = point
                print("FARTHEST POINT BHVR", time.time())
                return farthest_point
        except ValueError:
            print("No valid point")
            self.pub_state.publish(1)
            return None

    def get_vector(self, point1, point2, point3):
        """
        Calculate the Euclidean distance between two points.
        """
        distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

        # Get angle between vehicle and wp in odom frame
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        theta_radians = math.atan2(delta_y, delta_x)
        direction_from_vx = math.degrees(theta_radians)

        delta_x_relative = point3[0] - point2[0]
        delta_y_relative = point3[1] - point2[1]
        theta_radians_relative = math.atan2(delta_y_relative, delta_x_relative)
        direction_relative = math.degrees(theta_radians_relative)


        return distance, direction_from_vx, direction_relative

    def corner_behaviour(self, number_of_points):
        #init lists
        corner_bhvr_points = []
        points_in_odom_frame = []
        
        #The center point of the circle in vx frame
        point_of_obstacle = [self.distance_in_meters, -self.distance_in_meters]
        
        #list of angle increments
        angles = np.linspace(math.pi/2, 0, number_of_points)
        
        #list of circle points in vx_frame
        corner_bhvr_points = [(point_of_obstacle[0] + self.distance_in_meters * math.cos(angle), 
                            point_of_obstacle[1] + self.distance_in_meters * math.sin(angle))
                            for angle in angles]
        
        #Tf to odom frame
        for points in corner_bhvr_points:
            point_msg = PointStamped()
            point_msg.point.x = points[0]
            point_msg.point.y = points[1]
            point_in_odom_frame = tf2_geometry_msgs.do_transform_point(point_msg, self.base_to_odom_tf)
            points_in_odom_frame.append(point_in_odom_frame)

        return points_in_odom_frame

if __name__ == "__main__":
    rospy.init_node("waypoint_administrator")
    Wp_Admin()
    rospy.spin()