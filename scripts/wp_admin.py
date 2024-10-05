#!/usr/bin/env python3

#Author: Tony Jacob
#Part of RISE Project. 
#Assigns waypoints from the path to be fed to the vx nav system.
#tony.jacob@uri.edu

#rosbag record /alpha_rise/path/current_state /alpha_rise/path/distance_to_obstacle /alpha_rise/odometry/filtered/local
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PolygonStamped, Point32, PointStamped, Point
from std_msgs.msg import Float32
import math
from mvp_msgs.srv import GetStateRequest, GetState, ChangeState, ChangeStateRequest, GetWaypoints, GetWaypointsRequest
import time
import tf2_ros
import tf2_geometry_msgs
import numpy as np

class Wp_Admin:
    def __init__(self):
        """
        Constructer. Init all pubs, subs, variables
        """
        self.distance_in_meters = rospy.get_param("path_generator/standoff_distance_meters",15)
        update_waypoint_topic = rospy.get_param("waypoint_admin/update_waypoint_topic")
        path_topic = rospy.get_param("path_generator/path_topic")

        self.get_state_service = rospy.get_param("waypoint_admin/get_state_service", "/alpha_rise/helm/get_state")
        self.change_state_service = rospy.get_param("waypoint_admin/change_state_service", "/alpha_rise/helm/change_state")
        self.get_waypoint_service = rospy.get_param("waypoint_admin/get_waypoint", "/alpha_rise/helm/path_3d/get_next_waypoints")
        self.n_points = rospy.get_param("path_generator/points_to_sample_from_curve",20)

        #Waypoint selection param
        self.update_rate = rospy.get_param("waypoint_admin/check_state_update_rate")
        self.depth = rospy.get_param("waypoint_admin/z_value")

        self.search_mode_timer_param = rospy.get_param("waypoint_admin/search_mode_timer")
        
        #Declare Pubs
        self.pub_update = rospy.Publisher(update_waypoint_topic, PolygonStamped, queue_size=1)
        self.pub_state = rospy.Publisher(path_topic + '/state', Float32, queue_size=1)
        
        #Declare Subs
        rospy.Subscriber(path_topic, Path, callback=self.path_cB)
        rospy.Subscriber(path_topic + '/distance_to_obstacle', Float32, callback= self.distance_cB)
        rospy.Subscriber(path_topic +"/best_point", Point, callback=self.point_cB)

        #Declare variables
        self.check_state_timer = time.time()
        self.state = None
        self.distance_to_obstacle = None

        self.tf_buffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(self.tf_buffer)

        self.node_name = rospy.get_name()

        self.poses = []

        self.count_concentric_circles = 0

        self.search_mode_timer = time.time()

    def point_cB(self, msg):
        """
        Best point callback.
        """
        self.x, self.y, z = msg.x, msg.y, msg.z

    def distance_cB(self, msg):
        """
        Distance to obstacle callback. 
        This function syncs with the autonomy state machine indicator.
        Depending on the waypoints in the Helm, 
        One can determine if in Following (0) or IceReac (1)
        """
        self.distance_to_obstacle = msg.data
        
        get_waypoint_client = rospy.ServiceProxy(self.get_waypoint_service, GetWaypoints)
        request = GetWaypointsRequest()
        response = get_waypoint_client(request)
        # print(len(response.wpt))
        # 2 is the len(response.wpt) when in following.
        if len(response.wpt) == 2:
            self.pub_state.publish(0)
        else:
            self.pub_state.publish(1)

    def path_cB(self, msg):
        """
        Path topic callback.
        Depending on the number & value of points and state of the autonomy;
        The behaviours are implemented.
        """

        #Vx position and bearing in Odom frame.
        self.base_to_odom_tf = self.tf_buffer.lookup_transform("alpha_rise/odom", "alpha_rise/base_link", 
                                                               rospy.Time(0), rospy.Duration(1.0))
        
        #Odom frame point in Vx frame
        self.odom_to_base_tf = self.tf_buffer.lookup_transform("alpha_rise/base_link", "alpha_rise/odom", 
                                                               rospy.Time(0), rospy.Duration(1.0))
        #Create Waypoint Message
        wp = PolygonStamped()
        wp.header.stamp = msg.header.stamp
        wp.header.frame_id = msg.header.frame_id
        
        #Check state whether survey_3d or start
        self.check_state()

        # Valid path is n_points long. 
        # Path is always being published.
        # If same path, then no new path is then published.
        if len(msg.poses) == self.n_points:
            if self.state == "survey_3d":
                if msg.poses != self.poses:
                    rospy.loginfo("Following Mode")
                    wp.polygon.points.append(Point32(self.x ,self.y, self.depth))
                    self.pub_update.publish(wp)
                    self.poses = msg.poses
            
            #Iceberg Reacquisition Mode is when 
            #the vehicle reaches end of a valid path.
            elif self.state == "start":     
                self.iceberg_reacquisition_mode(wp)
        
        #Path is still published when no costmap. But the n_points is 1 (vx_x, vx_y)
        #We use that parameter to create a new bhvr mode.
        else:
            # rospy.loginfo("Searching Mode")
            if self.state == "start":
                self.count_concentric_circles += 1
                self.searching_mode(wp)

            elif self.state == "survey_3d":
                #If timer runs out, then the node is killed.
                if(time.time() - self.search_mode_timer) > self.search_mode_timer_param: #sec
                    service_client_change_state = rospy.ServiceProxy(self.change_state_service, ChangeState)
                    request = ChangeStateRequest("start", self.node_name)
                    response = service_client_change_state(request)
                    rospy.signal_shutdown("Search Mode Took too long")

    def searching_mode(self, wp):
        """
        Function to navigate the vehicle 
        to start searching for iceberg at depth.
        """

        if self.count_concentric_circles == 1:
            #Get vehicle position in Odom
            self.search_mode_center = PointStamped()
            self.search_mode_center.point.x = self.base_to_odom_tf.transform.translation.x
            self.search_mode_center.point.y = self.base_to_odom_tf.transform.translation.y

        #Transform to current vx_frame
        center_in_vx_frame = tf2_geometry_msgs.do_transform_point(self.search_mode_center, self.odom_to_base_tf)        
        
        #Grow the search radius incrementally.
        search_mode_radius = 10 * self.count_concentric_circles
        search_mode_points = self.draw_arc(number_of_points=self.n_points, 
                                            start_angle=0, 
                                            end_angle=2*math.pi,
                                            center=[center_in_vx_frame.point.x, center_in_vx_frame.point.y],
                                            radius = search_mode_radius)
    
        for i in range(len(search_mode_points)):
            wp.polygon.points.append(Point32(search_mode_points[i].point.x ,search_mode_points[i].point.y, self.depth))

        service_client_change_state = rospy.ServiceProxy(self.change_state_service, ChangeState)
        request = ChangeStateRequest("survey_3d", self.node_name)
        response = service_client_change_state(request)
        self.pub_update.publish(wp)
        time.sleep(1)

    def iceberg_reacquisition_mode(self, wp):
        """
        Function to navigate the vehicle 
        so as to reacquire acoustic contact
        """
        #Switch state to survey_3d
        service_client_change_state = rospy.ServiceProxy(self.change_state_service, ChangeState)
        request = ChangeStateRequest("survey_3d", self.node_name)
        response = service_client_change_state(request)
        #The center point of the circle in vx frame
        point_of_obstacle = [self.distance_in_meters, -self.distance_in_meters]
        corner_bhvr_points = self.draw_arc(number_of_points=self.n_points, 
                                                   start_angle=math.pi/2, 
                                                   end_angle=0,
                                                   center=point_of_obstacle,
                                                   radius = self.distance_in_meters)
    
        #Append the waypoints
        for i in range(len(corner_bhvr_points)):
            wp.polygon.points.append(Point32(corner_bhvr_points[i].point.x ,corner_bhvr_points[i].point.y, self.depth))
        rospy.loginfo("Iceberg Reacquisition Mode")
        self.pub_update.publish(wp)

    def check_state(self):
        """
        Function to check the state of the helm
        """
        elapsed_time = time.time() - self.check_state_timer
        #Check state every 2s
        if elapsed_time > self.update_rate:
            self.check_state_timer = time.time()
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

    def draw_arc(self, number_of_points, start_angle, end_angle, center, radius):
        """
        Create an arc in vehicle frame using parametric equation of circle.
        Then transform to odom.
        """
        #init lists
        corner_bhvr_points = []
        points_in_odom_frame = []
        
        #list of angle increments
        angles = np.linspace(start_angle, end_angle, number_of_points)

        #list of circle points in vx_frame
        corner_bhvr_points = [(center[0] + radius * math.cos(angle), 
                            center[1] + radius * math.sin(angle))
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