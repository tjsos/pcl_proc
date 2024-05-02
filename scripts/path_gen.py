#!/usr/bin/env python3

#Author: Tony Jacob
#Part of RISE Project. 
#Generates curve, standoff curve of the iceberg from a costmap
#tony.jacob@uri.edu

#rosbag record /vx_image /alpha_rise/odometry/filtered/local

import rospy
import tf2_ros
import tf.transformations as tf_transform
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
import scipy

class PathGen:
    def __init__(self) -> None:
        #Get params
        self.odom_frame = rospy.get_param("path_generator/odom_frame","alpha_rise/odom")
        self.base_frame = rospy.get_param("path_generator/base_frame","alpha_rise/base_link")

        #Debug
        self.debug = rospy.get_param("path_generator/debug",False)

        costmap_topic = rospy.get_param("path_generator/costmap_topic","/move_base/local_costmap/costmap")
        path_topic = rospy.get_param("path_generator/path_topic","/path")
        vx_image_topic = rospy.get_param("path_generator/vehicle_frame_image_topic","/vx_image")

        self.canny_min = rospy.get_param("path_generator/canny_min_threshold",200)
        self.canny_max = rospy.get_param("path_generator/canny_max_threshold",255)

        self.distance_in_meters = rospy.get_param("path_generator/standoff_distance_meters",15)
        self.n_points = rospy.get_param("path_generator/points_to_sample_from_curve",20)
        self.slope_tolerance = rospy.get_param("path_generator/slope_tolerance",1)
        
        #Costmap subscriber.
        rospy.Subscriber(costmap_topic, OccupancyGrid, self.mapCB)
        
        #Path Publisher
        self.pub = rospy.Publisher(path_topic, Path, queue_size=1)
        self.pub_slope = rospy.Publisher("/path/slope", Int16, queue_size=1)
        
        #Vx_frame image publisher
        self.image_pub = rospy.Publisher(vx_image_topic, Image, queue_size=1)
        self.bridge = CvBridge()
        
        #TF Buffer and Listener
        self.buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(self.buffer)

        #TF BroadCaster
        self.br = tf2_ros.TransformBroadcaster()
        
        #path placeholder
        self.poses = []
        
    def mapCB(self, message):
        """
        Transform Stuff
        """
        # Check robot_localisation has started publishing tf
        self.buffer.can_transform(self.base_frame, self.odom_frame, rospy.Time(), rospy.Duration(8.0))
        
        #Get Odom->Vx TF
        odom_vx_tf = self.buffer.lookup_transform(self.base_frame, self.odom_frame, rospy.Time())
        self.vx_yaw = tf_transform.euler_from_quaternion([odom_vx_tf.transform.rotation.x,
                                            odom_vx_tf.transform.rotation.y,
                                            odom_vx_tf.transform.rotation.z,
                                            odom_vx_tf.transform.rotation.w])[2]
        #Get Vx->Odom TF
        vx_odom_tf = self.buffer.lookup_transform(self.odom_frame, self.base_frame, rospy.Time())
        self.vx_x = vx_odom_tf.transform.translation.x
        self.vx_y = vx_odom_tf.transform.translation.y

        """
        Costmap
        """
        self.width = message.info.width
        self.height = message.info.height
        data = message.data
        self.frame = message.header.frame_id
        self.time = message.header.stamp

        self.resolution = message.info.resolution

        # # Broadcast Odom-> Costmap TF
        # odom_costmap_tf = TransformStamped()
        # odom_costmap_tf.header.stamp = rospy.Time.now()
        # odom_costmap_tf.header.frame_id = 'alpha_rise/odom'
        # odom_costmap_tf.child_frame_id = 'alpha_rise/costmap'
        # odom_costmap_tf.transform.translation.x = message.info.origin.position.x
        # odom_costmap_tf.transform.translation.y = message.info.origin.position.y
        # odom_costmap_tf.transform.translation.z = 0
        # odom_costmap_tf.transform.rotation.x = 0
        # odom_costmap_tf.transform.rotation.y = 0
        # odom_costmap_tf.transform.rotation.z = 0
        # odom_costmap_tf.transform.rotation.w = 1
        # self.br.sendTransform(odom_costmap_tf)

        # # Broadcast Costmap-> Costmap Image TF
        # costmap_costmap_image_tf = TransformStamped()
        # costmap_costmap_image_tf.header.stamp = rospy.Time.now()
        # costmap_costmap_image_tf.header.frame_id = 'alpha_rise/costmap'
        # costmap_costmap_image_tf.child_frame_id = 'alpha_rise/costmap/image'
        # # Width(cells) * resolution (meters/cell) = meters.
        # costmap_costmap_image_tf.transform.translation.x = self.width * self.resolution
        # costmap_costmap_image_tf.transform.translation.y = self.height * self.resolution
        # costmap_costmap_image_tf.transform.translation.z = 0
        # costmap_costmap_image_tf.transform.rotation.x = 0.7071
        # costmap_costmap_image_tf.transform.rotation.y = -0.7072
        # costmap_costmap_image_tf.transform.rotation.z = 0
        # costmap_costmap_image_tf.transform.rotation.w = 0

        # self.br.sendTransform(costmap_costmap_image_tf)

        
        """
        Image
        """
        # Costmap to Image
        data = np.array(data).reshape(self.width,self.height)
        data = data.astype(np.uint8)

        # Corrected for Costmap FRAME -> Image frame.
        data = cv2.flip(data, 1)  
        data = cv2.rotate(data, cv2.ROTATE_90_CLOCKWISE)

        # Dilate Raw Measurements to get solid reading       
        dilate = cv2.dilate(data, (5,5), 2)
        dilate = cv2.dilate(dilate, (5,5), 2)
        dilate = cv2.dilate(dilate, (5,5), 2)
        dilate = cv2.dilate(dilate, (5,5), 2)
        dilate = cv2.dilate(dilate, (5,5), 2)


        """
        Edges, Lines and Curves
        """
        #Get Edge
        canny_image = cv2.Canny(dilate,self.canny_min,self.canny_max)
        #Find cordinates of the edges
        raw_pixels = self.find_cordinates_of_max_value(canny_image)
        #Get usable edges
        edge, edge_polar = self.get_usable_edges(raw_pixels)
        #Image and points in Vx_frame
        vx_frame_image, vx_frame_points = self.plot_circles_vx_frame(edge_polar)
        #Standoff
        x_list, y_list, debug_image, path_frame_debug = self.edge_shifting(vx_frame_image)
        # Fit the line
        path_cells = self.curve_fit(x_list, y_list)
        # Project line in odom frame
        odom_frame_path = self.vx_to_odom_frame_tf(path_cells)
        """
        Path
        """
        self.pub_path(odom_frame_path)
        vx_frame_image_msg = self.bridge.cv2_to_imgmsg(vx_frame_image)
        self.image_pub.publish(vx_frame_image_msg)

        """
        Visualization
        """
        if self.debug:
        ## Compare the Canny and Custom
            compare_path = self.compare_two_lists(raw_pixels, odom_frame_path)
            viz_edges = self.compare_two_lists(raw_pixels, edge)
            compare_edge = self.compare_edge(vx_frame_image, path_cells)

            mix= np.hstack((data, viz_edges, compare_path, compare_edge))
            cv2.imshow("window", mix)
            cv2.imshow("debug", debug_image)
            cv2.imshow("path_frame", path_frame_debug)
            cv2.waitKey(1)

    
    def model_f(self,x, a, b, c):
        """
        The polynomial used to fit the points.
        """
        return a*x**2 + b* x**1 +c
        
    def edge_shifting(self, vx_frame:np.array):
        vx_frame_cropped = vx_frame[ :, self.width//2 :]       
        """        
        ------>xVEHICLE COSTMAP IMAGE
        |
        |
        |
       yV
        """
        self.new_origin = []
        img_cropped_cart = self.find_cordinates_of_max_value(vx_frame_cropped)

        if self.debug:
            debug = np.zeros((self.height, self.width//2,3), dtype=np.uint8)
            path_frame_debug = np.zeros((self.height, self.width,3), dtype=np.uint8)
        
            for coordinates in img_cropped_cart:
                center = tuple(coordinates)
                cv2.circle(debug, center, 1, (255,255,255), 1)
            
        if len(img_cropped_cart) > 2:
            #New origin from list of points.
            x_coordinates, y_coordinates = zip(*img_cropped_cart)
            
            
            min_y = min(y_coordinates)
            for pair in img_cropped_cart:
                # Check if the first element of the pair matches the given x_value
                if pair[1] == min_y:
                    # Return the corresponding y value
                    min_x = pair[0]
            self.new_origin.append(min_x)
            self.new_origin.append(min_y)
            if self.debug:
                cv2.circle(debug, (min_x, min_y), 1, (0,255,0), 1)
            #VX_FRAME_IMAGE -> PATH FRAME
            x_path_frame_list = []
            y_path_frame_list = []

            for coords in img_cropped_cart:
                shifted_x = coords[0] - self.new_origin[0]
                shifted_y = coords[1] - self.new_origin[1]
                x_path_frame_list.append(shifted_x)
                y_path_frame_list.append(shifted_y)
            
            """        
            ------>x VEHICLE COSTMAP IMAGE
            |         ->x PATH FRAME
            |        yV
            |
            yV
            """

            #LINEAR REGESH
            self.slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_path_frame_list, y_path_frame_list)
            
            if self.debug:
                lin_regress_debug = [round((self.slope) * x  + intercept) for x in x_path_frame_list]
    
                for cords in zip(x_path_frame_list,y_path_frame_list ):
                    center = tuple(cords)
                    cv2.circle(path_frame_debug, center, 1, (255,255,255), 1)
                for cords in zip(x_path_frame_list, lin_regress_debug):
                    center = tuple(cords)
                    cv2.circle(path_frame_debug, center, 1, (0,0,255), 1)

            self.slope = -(self.slope)
            
            # print(f"Slope {self.slope}, Yaw {math.degrees(self.vx_yaw)}")
            
            lingres  = [[x, y] for x, y in zip(x_path_frame_list, y_path_frame_list)]
            #PATH_FRAME -> BEST_FIT_LINE_FRAME
            x_line_frame_list, y_line_frame_list = self.rotate_points(lingres, math.degrees(math.atan((self.slope))))
            """        
            ------>xCOSTMAP IMAGE
            |         ->x LINE FRAME    
            |        yV
            |
            yV
            """
            distance_in_cells = self.distance_in_meters*1/self.resolution
            
            #Slope can have same values but different signs. Resulting in either shifting towards or away.
            if self.slope <= 0.0:
                y_line_frame_list = [y + distance_in_cells for y in y_line_frame_list]
            else:
            #SHIFT BY Y Standoff
                y_line_frame_list = [y - distance_in_cells for y in y_line_frame_list]
            
        else:
            x_line_frame_list = y_line_frame_list = 0
        if self.debug:
            return x_line_frame_list, y_line_frame_list, debug, path_frame_debug
        else:
            return x_line_frame_list, y_line_frame_list, None, None
    
    def rotate_points(self, points, angle_degrees):
        # Convert angle from degrees to radians
        angle_radians = math.radians(angle_degrees)
        
        # Define the rotation matrix
        rotation_matrix = [
            [math.cos(angle_radians), -math.sin(angle_radians)],
            [math.sin(angle_radians), math.cos(angle_radians)]
        ]
        
        # Initialize a list to store rotated points
        x_list = []
        y_list = []

        
        # Apply rotation to each point
        for point in points:
            # Perform matrix multiplication to rotate the point
            rotated_x = rotation_matrix[0][0] * point[0] + rotation_matrix[0][1] * point[1]
            rotated_y = rotation_matrix[1][0] * point[0] + rotation_matrix[1][1] * point[1]
            
            # Append rotated point to the list
            x_list.append((rotated_x))
            y_list.append((rotated_y))
        
        return x_list, y_list
       
    def curve_fit(self, x_coordinates:list, y_coordinates:list):
        """
        Fit the curve


        Args:
            vx_frame: Image depicting points in vehicle frame.
        
        Returns:
            sampled_coordinates: List of fitted points in vehicle frame
            polar_coordinates: List of fitted points in polar system with origin being center of image.
        """
        try:
            #FIT THE CURVE
            p_opt, p_cov = scipy.optimize.curve_fit(self.model_f, x_coordinates, y_coordinates)
            """        
            ------>xCOSTMAP IMAGE
            |         ->x PATH FRAME
            |        yV
            |
            yV
            """
            # print(p_opt)
            a_opt, b_opt, c_opt = p_opt
            #GET LINE
            x_model = np.linspace(min(x_coordinates), max(x_coordinates), self.n_points)
            y_model = self.model_f(x_model, a_opt, b_opt, c_opt)
            #LINE FRAME->PATH_FRAME
            lingres  = [[x, y] for x, y in zip(x_model, y_model)]
            x_path_frame_list, y_path_frame_list = self.rotate_points(lingres, math.degrees(math.atan(-(self.slope))))
            """        
            ------>xCOSTMAP IMAGE
            |         ->x PATH FRAME
            |        yV
            |
            yV
            """
            #PATH_FRAME -> VX_FRAME_IMAGE
            vx_frame_model = []
            path_frame = [[x, y] for x, y in zip(x_path_frame_list, y_path_frame_list)]

            for coords in path_frame:
                shifted_x = coords[0] + self.new_origin[0]
                shifted_y = coords[1] + self.new_origin[1]
                vx_frame_model.append([round(shifted_x), round(shifted_y)])
            """        
            ------>xCOSTMAP IMAGE
            |
            |
            |
            yV
            """
            # print(vx_frame_model)
            # vx_frame_model = [[x+ self.width//2, y] for x,y in path_frame]

            vx_frame_model = [[x+ self.width//2, y] for x,y in vx_frame_model]
            return vx_frame_model

        except TypeError:
            print("No solution found")
            return -1


    def vx_to_odom_frame_tf(self, cart_list:list):
        """
        Transfer it to odom frame.

        
        Args:
            polar_list: List of fitted points in polar system with origin being center of image

        Returns:
            cartesian_coordinates: List of fitted points in Odom frame
        """
        if cart_list == -1:
            return -1
        else:
            cartesian_coordinates = [(x - self.height//2, self.width//2 - y) for x, y in cart_list]
            """
            ------>x COSTMAP IMAGE
            |
            |   ----->x'
            |   |
            |   |
            |   Vy'
            yV
            """
            cartesian_coordinates = [(-y,-x) for x, y in cartesian_coordinates]
            """
            ------>x COSTMAP IMAGE
            |
            |      ^x
            |      |
            |      |
            |  y<--- VEHICLE FRAME
            |   
            yV
            """
            # Step 2: Convert shifted Cartesian coordinates to polar coordinates
            polar_coordinates = [[np.sqrt(x**2 + y**2), (np.arctan2(y, x))] for x, y in cartesian_coordinates]
            
            vx_theta = self.vx_yaw
            
            # if vx_theta > 1.0:
            #     vx_theta = 1.0
            # # Discout vx_yaw
            for points in polar_coordinates:
                points[1] = (points[1] + np.float64(vx_theta))
    
            # Convert to Cartesian (x,y)
            cartesian_coordinates = [[round(r * np.cos(theta)), round(r * np.sin(theta))] for r, theta in polar_coordinates]
            """
            ------>x COSTMAP IMAGE
            |
            |      ^x
            |      |
            |      |
            |  y<--- VEHICLE FRAME
            |   
            yV
            """
            cartesian_coordinates = [(-y,-x) for x, y in cartesian_coordinates]
            """
            ------>x COSTMAP IMAGE
            |
            |   ----->x'
            |   |
            |   |
            |   Vy'
            yV
            """
            # Shift back to Image frame
            cartesian_coordinates = [[int(x+self.width//2),int(self.height//2 - y)] for x, y in cartesian_coordinates]
            """        
            ------>xCOSTMAP IMAGE
            |
            |
            |
            yV
            """
            # cartesian_coordinates = sorted(cartesian_coordinates, key=lambda coord: coord[1], reverse=True)

            return cartesian_coordinates
        
    def find_cordinates_of_max_value(self, image:np.array):
        """
        Find cordinates of white pixels

        Args:
            image: Image after Canny edge detection.
        
        Returns:
            max_intensity_coordinates: [[x,y]] of edges.
        """
        max_intensity = np.max(image)
        # Get the coordinates of pixels with the maximum intensity
        max_intensity_coordinates = np.column_stack(np.where(image == max_intensity))
        # column_stack GIVES INVERTED [x,y] from that of image frame
        max_intensity_coordinates = [[y,x] for x,y in max_intensity_coordinates]
        return max_intensity_coordinates
    
    def get_usable_edges(self, coordinates:list):
        """
        Use it to find the outer edge from the canny image. 
        The resultant is the cordinates of max likelihood where the obstacle is.

        Args:
            coordinates: [[x,y]] of edges from Image.

        Returns:
            cartesian_coordinates: List of points that are on the outer edge from canny image.
            polar_coordinates: List of points [r, theta] from the center of the image. (Odom frame, vx as origin)
        
        ------>xCOSTMAP IMAGE
        |
        |
        |
       yV
        """
        polar_dict = {}

        # Step 1: Shift coordinates to center of the image
        shifted_coordinates = [(x - self.height//2, self.width//2 - y) for x, y in coordinates]
        """
        ------>x COSTMAP IMAGE
        |
        |   ----->x'
        |   |
        |   |
        |   Vy'
       yV
        """
        shifted_coordinates = [(-y,-x) for x, y in shifted_coordinates]
        """
        ------>x COSTMAP IMAGE
        |
        |      ^x
        |      |
        |      |
        |  y<--- VEHICLE IMAGE FRAME
        |   
       yV
        """
        # Step 2: Convert shifted Cartesian coordinates to polar coordinates
        polar_coordinates = [(np.sqrt(x**2 + y**2), round(np.arctan2(y, x),1)) for x, y in shifted_coordinates]
        
        # Step 3: Check for duplicate theta, keep the smallest distance
        for r, theta in polar_coordinates:
            if theta in polar_dict:
                # If a duplicate theta is encountered, keep the point with the shortest distance
                if r < polar_dict[theta][0]:
                    polar_dict[theta] = (r, theta)
            else:
                polar_dict[theta] = (r, theta)
        polar_coordinates = list(polar_dict.values())
        # Step 4: Convert to Cartesian. (x, y)
        cartesian_coordinates = [[round(r * np.cos(theta)), round(r * np.sin(theta))] for r, theta in polar_coordinates]
        """
        ------>x COSTMAP IMAGE
        |
        |      ^x
        |      |
        |      |
        |  y<--- VEHICLE IMAGE FRAME
        |   
       yV
        """
        cartesian_coordinates = [(-y,-x) for x, y in cartesian_coordinates]
        """
        ------>x COSTMAP IMAGE
        |
        |   ----->x'
        |   |
        |   |
        |   Vy'
       yV
        """

        # Step 5: Shift back
        cartesian_coordinates = [[int(x+self.width//2),int(self.height//2 - y),] for x, y in cartesian_coordinates]
        """ 
        ------>xCOSTMAP IMAGE
        |
        |
        |
       yV
        """
       # Step 6: sample //TO-DO; Time based sampling.
        # if len(cartesian_coordinates) > self.width//4:
        #     diff = len(cartesian_coordinates) - self.width//4
        #     for i in range(diff):
        #         cartesian_coordinates.pop(i)
        #
        # POLAR_CORDINATES IS IN VEHICLE FRAME
        # [x,y] [r,theta]
        return cartesian_coordinates, polar_coordinates
    
    def plot_circles_vx_frame(self, coordinates_list:list, radius=1, color=255):
        """
        Viz function with to show vehicle frame. What the vehicle sees is constant in this frame

        Args:
            coordinates_list: List of points in polar system in Odom frame.

        Returns:
            image: Image with points plotted in vehicle frame.
            cartesian_coordinates: list of points in vehicle frame
        """
        yawd_p = [list(t) for t in coordinates_list]

        vx_theta = self.vx_yaw

        for points in yawd_p:
            points[1] = points[1] - np.float64(vx_theta)
            

        # Convert to Cartesian.
        cartesian_coordinates = [[round(r * np.cos(theta)), round(r * np.sin(theta))] for r, theta in yawd_p]
        """
        ------>x COSTMAP IMAGE
        |
        |      ^x
        |      |
        |      |
        |  y<--- VEHICLE FRAME
        |   
       yV
        """
        cartesian_coordinates = [(-y,-x) for x, y in cartesian_coordinates]
        """
        ------>x COSTMAP IMAGE
        |
        |   ----->x'
        |   |
        |   |
        |   Vy'
       yV
        """
        # Shift back to Image frame
        cartesian_coordinates = [[int(x+self.width//2),int(self.height//2 - y)] for x, y in cartesian_coordinates]
        """ 
        ------>xCOSTMAP IMAGE
        |
        |
        |
       yV
        """

        # Create an empty image (numpy array)
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        # Iterate through the
        #  list of coordinates and draw circles
        for coordinates in cartesian_coordinates:
            center = tuple(coordinates)
            cv2.circle(image, center, radius, color, 1)

        # image = cv2.line(image, (0, 3*self.width//4), (200, 3*self.width//4), 100, 1)
        # image = cv2.line(image, (100, 0), (100, 200), 100, 1)

        #Draw Vx frame
        # image = cv2.line(image, (self.width//2, self.height//2), (self.width//2, self.height//2-5), 200, 1)
        # image = cv2.line(image, (self.width//2, self.height//2), (self.width//2-5, self.height//2), 200, 1) 

        return image, cartesian_coordinates
    
    def compare_edge(self, frame, points_list):
        for coordinates in points_list:
            center = tuple(coordinates)
            cv2.circle(frame, center, 1, 100, 1)
        return frame
    
    def compare_two_lists(self, list1, list2):
        """
        Viz function to plot the original canny and extracted edge

        
        Args:
            list1: List of points (raw_pixels)
            list2: List of points

        Returns:
            image: Image with points plotted.
        """
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        # Iterate through the list of coordinates and draw circles
        
        for coordinates in list1:
            center = tuple(coordinates)
            cv2.circle(image, center, 1, 100)

        for coordinates in list2:
            center = tuple(coordinates)
            cv2.circle(image, center, 1, 255)
        
        return image
    
    def pub_path(self, shifted_coordinates):
        """
        Create and send the path message


        Args:
            shifted_coordinates: List of fitted points in odom frame
        """
        if type(shifted_coordinates) == int:
            return -1
    
        else:
            path = Path()
            self.path = Path()
            slope_msg = Int16()
            path.header.frame_id = self.frame#"alpha_rise/base_link"
            path.header.stamp =  self.time
            for index, cords in enumerate(shifted_coordinates):
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = self.frame#
                pose_stamped.header.stamp = self.time
                pose_stamped.header.seq = index

                #IMAGE TO MAP FRAME FIXED TO ODOMs
                """
                ------>x COSTMAP IMAGE
                |
                |      ^x
                |      |
                |      |
                |  y<--- VEHICLE FRAME
                |   
                yV
                """
                x = -cords[1]
                y = -cords[0] 

                x = ((self.height//2 + x) * self.resolution) + self.vx_x
                y = ((self.width//2 + y)* self.resolution ) + self.vx_y

                pose_stamped.pose.position.x = x#self.height//4 - cords[1] * self.resolution
                pose_stamped.pose.position.y = y#self.height//4 - cords[0] * self.resolution 
                pose_stamped.pose.orientation.x = 0
                pose_stamped.pose.orientation.y = 0
                pose_stamped.pose.orientation.z = 0
                pose_stamped.pose.orientation.w = 1
                path.poses.append(pose_stamped)
            #Send previous valid path if present path is invalid.
            if abs(self.slope) > self.slope_tolerance:
                print("acceptable slope")
                slope_msg.data = 1
                self.path = path
                self.poses = path.poses
                self.pub.publish(self.path)
                self.pub_slope.publish(slope_msg)
            else:
                print("unacceptable")
                slope_msg.data = 0
                self.path.header = path.header
                self.path.header.stamp = rospy.Time()
                self.path.poses = self.poses
                self.pub.publish(self.path)
                self.pub_slope.publish(slope_msg)
                
    
if __name__ == "__main__":
    rospy.init_node('path_node')
    l = PathGen()
    rospy.spin()