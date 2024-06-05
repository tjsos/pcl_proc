#!/usr/bin/env python3

#Author: Tony Jacob
#Part of RISE Project. 
#Generates curve, standoff curve of the iceberg from a costmap
#tony.jacob@uri.edu

import rospy
import tf2_ros
import tf.transformations as tf_transform
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float32
import numpy as np
import cv2
import math
import scipy
import path_utils

class PathGen:
    def __init__(self) -> None:
        #Get params
        self.odom_frame = rospy.get_param("path_generator/odom_frame","alpha_rise/odom")
        self.base_frame = rospy.get_param("path_generator/base_frame","alpha_rise/base_link")

        #Debug
        self.debug = rospy.get_param("path_generator/debug",False)

        costmap_topic = rospy.get_param("path_generator/costmap_topic")
        path_topic = rospy.get_param("path_generator/path_topic")
        obstacle_distance_topic = rospy.get_param("path_generator/distance_to_obstacle_topic")

        self.canny_min = rospy.get_param("path_generator/canny_min_threshold",200)
        self.canny_max = rospy.get_param("path_generator/canny_max_threshold",255)

        self.distance_in_meters = rospy.get_param("path_generator/standoff_distance_meters",15)
        self.n_points = rospy.get_param("path_generator/points_to_sample_from_curve",20)
        
        #Costmap subscriber.
        rospy.Subscriber(costmap_topic, OccupancyGrid, self.mapCB)
        
        #Path Publisher
        self.pub_path = rospy.Publisher(path_topic, Path, queue_size=1)

        #Vx_frame image publisher
        self.obstacle_distance_pub = rospy.Publisher(obstacle_distance_topic, Float32, queue_size=1)
        
        #TF Buffer and Listener
        self.buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(self.buffer)

        #TF BroadCaster
        self.br = tf2_ros.TransformBroadcaster()
        
        #path placeholder
        self.poses = []
        
    def mapCB(self, message:OccupancyGrid):
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
        raw_pixels = path_utils.find_cordinates_of_max_value(canny_image)
        #Get usable edges
        edge, edge_polar = self.get_usable_edges(raw_pixels)
        #Image and points in Vx_frame
        vx_frame_image, vx_frame_points = self.transform_to_vx_frame(edge_polar)
        #Insert Standoff
        x_list, y_list, edge_frame_debug = self.edge_shifting(vx_frame_image)
        # Fit the line
        path_cells = self.curve_fit(x_list, y_list)
        # Project line in odom frame
        path_odom_frame = self.vx_to_odom_frame_tf(path_cells)
        """
        Path
        """
        self.convert_and_publish_path(path_odom_frame)

        """
        Distance to Obstacle
        """
        distance_to_obstacle = self.get_distance_to_obstacle(vx_frame_image)
        self.obstacle_distance_pub.publish(distance_to_obstacle)

        """
        Visualization
        """
        if self.debug:
        ## Compare the Canny and Custom
            compare_path = path_utils. compare_two_lists(raw_pixels, path_odom_frame, self.height, self.width)
            viz_edges = path_utils.compare_two_lists(raw_pixels, edge, self.height, self.width)
            compare_edge = path_utils.compare_points_with_image(vx_frame_image, path_cells)

            mix= np.hstack((data, viz_edges,compare_edge, compare_path))
            cv2.imshow("window", mix)
            cv2.imshow("edge_frame", edge_frame_debug)
            cv2.waitKey(1)
    
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
    
    def transform_to_vx_frame(self, coordinates_list:list, radius=1, color=255):
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
    
    def edge_shifting(self, vx_frame:np.array):
        # vx_frame_cropped = vx_frame[ :, self.width//2 :]
        vx_frame_cropped = vx_frame[ :, :]       
               
        """        
        ------>xVEHICLE COSTMAP IMAGE
        |
        |
        |
       yV
        """
        self.new_origin = []
        vx_frame_pixels = path_utils.find_cordinates_of_max_value(vx_frame_cropped)

        #Atleast need 2 points
        if len(vx_frame_pixels) > 2:
            #New origin from list of points.
            x_coordinates, y_coordinates = zip(*vx_frame_pixels)
            
            #Edge frame origin is point with min y in image frame.
            min_y = min(y_coordinates)
            for pair in vx_frame_pixels:
                # Check if the first element of the pair matches the given x_value
                if pair[1] == min_y:
                    # Return the corresponding y value
                    min_x = pair[0]
            #New origin
            self.new_origin.append(min_x)
            self.new_origin.append(min_y)
            
            ##VX_FRAME_IMAGE -> EDGE FRAME
            x_edge_frame_list = []
            y_edge_frame_list = []
            for coords in vx_frame_pixels:
                shifted_x = coords[0] - self.new_origin[0]
                shifted_y = coords[1] - self.new_origin[1]
                x_edge_frame_list.append(shifted_x)
                y_edge_frame_list.append(shifted_y)
            
            #Vehicle cord in edge frame
            vx_x_edge_frame = self.width//2 - self.new_origin[0]
            vx_y_edge_frame = self.height//2 - self.new_origin[1]
            """        
            ------>x VEHICLE COSTMAP IMAGE
            |         ->x PATH FRAME
            |        yV
            |
            yV
            """

            ##LINEAR REGESH
            # self.slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_edge_frame_list, 
                                                                                    #   y_edge_frame_list)
            self.slope, intercept = path_utils.calculate_slope(x_edge_frame_list, y_edge_frame_list)

            #Slope Angle (in radian)
            self.angle_from_e_to_l = math.atan(self.slope)
            
            if self.debug:
                edge_frame_debug = np.zeros((self.height, self.width,3), dtype=np.uint8)

                lin_regress_debug = [round((self.slope) * x  + intercept) for x in x_edge_frame_list]
    
                for cords in zip(x_edge_frame_list,y_edge_frame_list ):
                    cords = list(cords)
                    cords[0] = cords[0]+self.width//2
                    cords[1] = cords[1]+self.height//2
                    center = tuple(cords)
                    cv2.circle(edge_frame_debug, center, 1, (255,255,255), 1)
                
                #draw best fit line
                for cords in zip(x_edge_frame_list, lin_regress_debug):
                    cords = list(cords)
                    cords[0] = cords[0]+self.width//2
                    cords[1] = cords[1]+self.height//2
                    center = tuple(cords)
                    cv2.circle(edge_frame_debug, center, 1, (0,0,255), 1)

                #Draw origin of Edge Frame
                cv2.circle(edge_frame_debug, (self.width//2, 
                                              self.height//2), 1, (0,255,0), 2)
                
            lingres  = [[x, y] for x, y in zip(x_edge_frame_list, y_edge_frame_list)]

            ##EDGE_FRAME -> LINE_FRAME
            x_line_frame_list, y_line_frame_list = path_utils.rotate_points(lingres, self.angle_from_e_to_l)
            #Vehicle cords in Line Frame
            vx_x_line_frame, vx_y_line_frame = path_utils.rotate_points([[vx_x_edge_frame, vx_y_edge_frame]], self.angle_from_e_to_l)

            """        
            ------>xCOSTMAP IMAGE
            |         ->x LINE FRAME    
            |        yV
            |
            yV
            """

            #Insert standoff distance
            distance_in_cells = self.distance_in_meters*1/self.resolution
            
            if round(vx_y_line_frame[0]) > 0.0:
                y_line_frame_list = [y + distance_in_cells for y in y_line_frame_list]
            else:
                y_line_frame_list = [y - distance_in_cells for y in y_line_frame_list]
            
        else:
            x_line_frame_list = y_line_frame_list = 0
        
        if self.debug:
            return x_line_frame_list, y_line_frame_list, edge_frame_debug
        else:
            return x_line_frame_list, y_line_frame_list, None
       
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
            p_opt, p_cov = scipy.optimize.curve_fit(path_utils.model_f, x_coordinates, y_coordinates)
            """        
            ------>xCOSTMAP IMAGE
            |         ->x PATH FRAME
            |        yV
            |
            yV
            """
            # print(p_opt)
            a_opt, b_opt, c_opt = p_opt
            
            ##GET CURVE
            x_model = np.linspace(min(x_coordinates), max(x_coordinates), self.n_points)
            y_model = path_utils.model_f(x_model, a_opt, b_opt, c_opt)
            
            ##LINE FRAME->EDGE_FRAME
            lingres  = [[x, y] for x, y in zip(x_model, y_model)]
            x_edge_frame_list, y_edge_frame_list = path_utils.rotate_points(lingres, -self.angle_from_e_to_l)
            """        
            ------>xCOSTMAP IMAGE
            |         ->x PATH FRAME
            |        yV
            |
            yV
            """
            ##EDGE_FRAME -> VX_FRAME_IMAGE
            vx_frame_model = []
            edge_frame = [[x, y] for x, y in zip(x_edge_frame_list, y_edge_frame_list)]

            for coords in edge_frame:
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

            vx_frame_model = [[x, y] for x,y in vx_frame_model]
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
    
    def convert_and_publish_path(self, shifted_coordinates:list):
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

                x = ((self.width//2 + x) * self.resolution) + self.vx_x
                y = ((self.height//2 + y)* self.resolution ) + self.vx_y

                pose_stamped.pose.position.x = x#self.height//4 - cords[1] * self.resolution
                pose_stamped.pose.position.y = y#self.height//4 - cords[0] * self.resolution 
                pose_stamped.pose.orientation.x = 0
                pose_stamped.pose.orientation.y = 0
                pose_stamped.pose.orientation.z = 0
                pose_stamped.pose.orientation.w = 1
                path.poses.append(pose_stamped)
            self.pub_path.publish(path)
    
    def get_distance_to_obstacle(self, vx_image):
        valid_points= []
        sum_distance =0
        raw_pixels = path_utils.find_cordinates_of_max_value(vx_image)
        shifted_coordinates = [(x - self.height//2, self.width//2 - y) for x, y in raw_pixels]
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
        |  y<--- VEHICLE FRAME
        |   
        yV
        """

        image_polar = [(np.sqrt(x**2 + y**2), np.degrees(np.arctan2(y, x))) for x,y in shifted_coordinates]

        for points in image_polar:
            if -135<points[1]<-45:
                valid_points.append(points)
        """
                ^
                | /
                |/
            <---
                 \
                  \
        """
        for points in valid_points:
            sum_distance += points[0]
        if len(valid_points)!= 0:
            return (sum_distance * self.resolution)/len(valid_points)
        else:
            return 0
        
if __name__ == "__main__":
    rospy.init_node('path_node')
    PathGen()
    rospy.spin()