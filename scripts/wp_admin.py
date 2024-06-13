#!/usr/bin/env python3

#Author: Tony Jacob
#Part of RISE Project. 
#Assigns waypoints from the path to be fed to the vx nav system.
#tony.jacob@uri.edu

#rosbag record /alpha_rise/path/current_state /alpha_rise/path/distance_to_obstacle /alpha_rise/odometry/filtered/local
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PolygonStamped, Point32, PointStamped
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
        self.get_state_service = rospy.get_param("waypoint_admin/get_state_service", "/alpha_rise/helm/get_state")
        self.change_state_service = rospy.get_param("waypoint_admin/change_state_service", "/alpha_rise/helm/change_state")
     
        #Declare Pubs
        self.pub_update = rospy.Publisher(update_waypoint_topic, PolygonStamped, queue_size=1)
        self.pub_append = rospy.Publisher(append_waypoint_topic, PolygonStamped, queue_size=1)
        self.pub_state = rospy.Publisher(state_topic, Float32, queue_size=1)
        
        #Declare Subs
        rospy.Subscriber(path_topic, Path, callback=self.path_cB)
        rospy.Subscriber(distance_to_obstacle_topic, Float32, callback= self.distance_cB)

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
        if self.state == "survey_3d":
            # if self.distance_to_obstacle > 12:
                goal = self.find_point_to_follow(msg)  
                if goal != None:
                    x_c = goal.pose.position.x
                    y_c = goal.pose.position.y
                    
                    wp.polygon.points.append(Point32(x_c ,y_c, 0))

                    self.pub_update.publish(wp)
                    self.pub_state.publish(1)
                else:
                    self.pub_state.publish(0)

        elif self.state == "start":
            self.pub_state.publish(0)
            
            #Switch state to survey_3d
            service_client_change_state = rospy.ServiceProxy(self.change_state_service, ChangeState)
            request = ChangeStateRequest("survey_3d", self.node_name)
            response = service_client_change_state(request)
            print(f"changed to {response.state.name}")
            corner_bhvr_points = self.corner_behaviour(number_of_points=20)
            
            #Append the waypoints
            for i in range(len(corner_bhvr_points)):
                wp.polygon.points.append(Point32(corner_bhvr_points[i].point.x ,corner_bhvr_points[i].point.y, 0))
                
            self.pub_update.publish(wp)

    def check_state(self):
        elapsed_time = time.time() - self.start_time
        #Check state every 2s
        if elapsed_time > 0.5:
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
        distance_of_valid_xy = []
        # Iterate through the poses in the path
        for pose_stamped in path_msg.poses:
            #Transform the points in vx frame
            point_in_vx_frame = tf2_geometry_msgs.do_transform_pose(pose_stamped, self.odom_to_base_tf)
            
            #Get distance and bearing to the point from the vehicle
            distance,direction = self.get_vector([0,0], [point_in_vx_frame.pose.position.x,
                                                         point_in_vx_frame.pose.position.y])
            #Filter out a sector. REP 103 convention. +ve is counterclockwise from vehicle nose.
            if -30 < direction and direction < 60:
                #Gather pose_msgs, xy, and distances of such points
                valid_pose.append(pose_stamped)
                valid_xy.append([point_in_vx_frame.pose.position.x,
                                point_in_vx_frame.pose.position.y])
                distance_of_valid_xy.append(distance)
        
        #Check for cross track deviation.
        all_y_less_than_threshold = all(y < 2 for x,y in valid_xy)

        #If all points are within desired deviation, follow the farthest point.
        if all_y_less_than_threshold:
            farthest_point = None
            max_distance = float('-inf')
            for index, point in enumerate(valid_pose):
                if distance_of_valid_xy[index] > max_distance:
                    max_distance = distance_of_valid_xy[index]
                    farthest_point = point
            print("farhest_pont", time.time())
            return farthest_point
        #If not, follow the farthest point with the lowest y_deviation.
        else:
            lowest_y_point = None
            min_y = float('inf')
            max_distance = float('-inf')
            for index,point in enumerate(valid_xy):
                if point[1] < min_y:
                    if distance_of_valid_xy[index] > max_distance:
                        min_y = point[1]
                        max_distance = distance_of_valid_xy[index]
                        lowest_y_point = valid_pose[index]
            print("lowest_y_point",time.time())
            return lowest_y_point
        
    def get_vector(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.
        """
        distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        # Get angle between vehicle and wp in odom frame
        theta_radians = math.atan2(delta_y, delta_x)
        direction = math.degrees(theta_radians)
        return distance, direction

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