#!/usr/bin/env python3

#Author: Tony Jacob
#Part of RISE Project. 
#Assigns waypoints from the path to be fed to the vx nav system.
#tony.jacob@uri.edu

#rosbag record /alpha_rise/path/current_state /alpha_rise/path/distance_to_obstacle /alpha_rise/odometry/filtered/local
import rospy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PolygonStamped, Point32, PointStamped, PoseStamped, Point
from std_msgs.msg import Float32
import tf.transformations as tf_transform
import math
from mvp_msgs.srv import GetStateRequest, GetState, ChangeState, ChangeStateRequest, GetWaypoints, GetWaypointsRequest
import time
import tf2_ros
import tf2_geometry_msgs
import numpy as np

class Wp_Admin:
    def __init__(self):
        #Get Params
        self.distance_in_meters = rospy.get_param("path_generator/standoff_distance_meters",15)
        update_waypoint_topic = rospy.get_param("waypoint_admin/update_waypoint_topic")
        path_topic = rospy.get_param("path_generator/path_topic")
        state_topic = path_topic + '/state'
        distance_to_obstacle_topic = path_topic + '/distance_to_obstacle'

        self.get_state_service = rospy.get_param("waypoint_admin/get_state_service", "/alpha_rise/helm/get_state")
        self.change_state_service = rospy.get_param("waypoint_admin/change_state_service", "/alpha_rise/helm/change_state")
        self.get_waypoint_service = rospy.get_param("waypoint_admin/get_waypoint", "/alpha_rise/helm/path_3d/get_next_waypoints")

        #Waypoint selection param
        self.update_rate = rospy.get_param("waypoint_admin/check_state_update_rate")
        self.depth = rospy.get_param("waypoint_admin/z_value")
        
        #Declare Pubs
        self.pub_update = rospy.Publisher(update_waypoint_topic, PolygonStamped, queue_size=1)
        self.pub_state = rospy.Publisher(state_topic, Float32, queue_size=1)
        
        #Declare Subs
        rospy.Subscriber(path_topic, Path, callback=self.path_cB)
        rospy.Subscriber(distance_to_obstacle_topic, Float32, callback= self.distance_cB)
        rospy.Subscriber(path_topic +"/best_point", Point, callback=self.point_cB)

        #Declare variables
        self.start_time = time.time()
        self.state = None
        self.distance_to_obstacle = None

        self.tf_buffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(self.tf_buffer)

        self.node_name = rospy.get_name()
    
    def point_cB(self, msg):
        #Get the best point
        self.x, self.y, z = msg.x, msg.y, msg.z

    def distance_cB(self, msg):
        #Depending on the waypoints in the Helm, 
        #One can determine if in Following (0) or IceReac (1)
        self.distance_to_obstacle = msg.data
        
        get_waypoint_client = rospy.ServiceProxy(self.get_waypoint_service, GetWaypoints)
        request = GetWaypointsRequest()
        response = get_waypoint_client(request)
        if len(response.wpt) == 2:
            self.pub_state.publish(0)
        else:
            self.pub_state.publish(1)

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
        if len(msg.poses) > 10:
            if self.state == "survey_3d":
                x_c = self.x 
                y_c = self.y
                print("[INFO]: Following Mode", time.time())
                wp.polygon.points.append(Point32(x_c ,y_c, self.depth))
                self.pub_update.publish(wp)

            elif self.state == "start":     
                self.no_path_bhvr(wp)
                
        else:
            #Else no path, iceberg acquisiton mode
            self.no_path_bhvr(wp)

    def no_path_bhvr(self, wp):
        if self.state == "start":
            #Switch state to survey_3d
            service_client_change_state = rospy.ServiceProxy(self.change_state_service, ChangeState)
            request = ChangeStateRequest("survey_3d", self.node_name)
            response = service_client_change_state(request)
            corner_bhvr_points = self.corner_behaviour(number_of_points=20)
        
            #Append the waypoints
            for i in range(len(corner_bhvr_points)):
                wp.polygon.points.append(Point32(corner_bhvr_points[i].point.x ,corner_bhvr_points[i].point.y, self.depth))
            print("[INFO]: Iceberg Reacquisition Mode", time.time())
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