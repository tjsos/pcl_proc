#!/usr/bin/env python3

#Author: Tony Jacob
#Part of RISE Project. 
#Generates curve, standoff curve of the iceberg from a costmap
#tony.jacob@uri.edu

import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
import cv2
import math
import scipy

class CostMapProcess:
    def __init__(self) -> None:
        #Path Publisher
        self.pub = rospy.Publisher("/path", Path, queue_size=1)

        #TF Buffer and Listener
        self.buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(self.buffer)
        #TF BroadCaster
        self.br = tf2_ros.TransformBroadcaster()

        #Costmap subscriber.
        rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.mapCB)

    def mapCB(self, message):
        """
        Transform Stuff
        """
        self.buffer.can_transform('alpha_rise/base_link', 'alpha_rise/odom', rospy.Time(), rospy.Duration(4.0))
        
        #Get Odom->Vx TF
        odom_vx_tf = self.buffer.lookup_transform('alpha_rise/base_link', 'alpha_rise/odom', rospy.Time())
        self.vx_yaw = euler_from_quaternion([odom_vx_tf.transform.rotation.x,
                                            odom_vx_tf.transform.rotation.y,
                                            odom_vx_tf.transform.rotation.z,
                                            odom_vx_tf.transform.rotation.w])[2]
        
        #Get Vx->Odom TF
        vx_odom_tf = self.buffer.lookup_transform('alpha_rise/odom', 'alpha_rise/base_link', rospy.Time())
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

        # Broadcast Odom-> Costmap TF
        odom_costmap_tf = TransformStamped()
        odom_costmap_tf.header.stamp = rospy.Time.now()
        odom_costmap_tf.header.frame_id = 'alpha_rise/odom'
        odom_costmap_tf.child_frame_id = 'alpha_rise/costmap'
        odom_costmap_tf.transform.translation.x = message.info.origin.position.x
        odom_costmap_tf.transform.translation.y = message.info.origin.position.y
        odom_costmap_tf.transform.translation.z = 0
        odom_costmap_tf.transform.rotation.x = 0
        odom_costmap_tf.transform.rotation.y = 0
        odom_costmap_tf.transform.rotation.z = 0
        odom_costmap_tf.transform.rotation.w = 1

        # Broadcast Costmap-> Costmap Image TF
        odom_costmap_image_tf = TransformStamped()
        odom_costmap_image_tf.header.stamp = rospy.Time.now()
        odom_costmap_image_tf.header.frame_id = 'alpha_rise/costmap'
        odom_costmap_image_tf.child_frame_id = 'alpha_rise/costmap/image'
        # Width(cells) * resolution (meters/cell) = meters.
        odom_costmap_image_tf.transform.translation.x = self.width * self.resolution
        odom_costmap_image_tf.transform.translation.y = self.height * self.resolution
        odom_costmap_image_tf.transform.translation.z = 0
        odom_costmap_image_tf.transform.rotation.x = 0.7071
        odom_costmap_image_tf.transform.rotation.y = -0.7072
        odom_costmap_image_tf.transform.rotation.z = 0
        odom_costmap_image_tf.transform.rotation.w = 0

        self.br.sendTransform(odom_costmap_tf)
        self.br.sendTransform(odom_costmap_image_tf)

        
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
        canny_image = cv2.Canny(dilate,200,255)
        #Find cordinates of the edges
        raw_pixels = self.find_cordinates_of_max_value(canny_image)
        #Get usable edges
        edge, edge_polar = self.get_usable_edges(raw_pixels)
        #Image and points in Vx_frame
        vx_frame_image, vx_frame_points = self.plot_circles_vx_frame(edge_polar)
        # Fit the line
        path_cells, path_cells_polar = self.curve_fit(vx_frame_image)
        # Project line in odom frame
        odom_frame_path = self.vx_to_odom_frame_tf(path_cells_polar)

        """
        Path
        """
        self.pub_path(odom_frame_path)

        """
        Visualization
        """

        ## Compare the Canny and Custom
        compare_path = self.compare_two_lists(raw_pixels, odom_frame_path)
        compare_edges = self.compare_two_lists(raw_pixels, edge)

        mix= np.hstack((data, compare_edges, compare_path, vx_frame_image))
        cv2.imshow("window", mix)
        cv2.waitKey(1)
    
    def model_f(self,x, a, b, c, d):
        """
        The polynomial used to fit the points.
        """
        # return a*(x-b)**2 + c
        return a*x**3 + b* x**2 +c*x +d
        
    def curve_fit(self, vx_frame):
        """
        Fit the curve


        Args:
            vx_frame: Image depicting points in vehicle frame.
        
        Returns:
            sampled_coordinates: List of fitted points in vehicle frame
            polar_coordinates: List of fitted points in polar system with origin being center of image.
        """
        vx_frame_cropped = vx_frame[ :, :]
        img_cropped_cart = self.find_cordinates_of_max_value(vx_frame_cropped)
        if img_cropped_cart is not []:
            x_coordinates, y_coordinates = zip(*img_cropped_cart)
            try:
                #FIT THE CURVE
                p_opt, p_cov = scipy.optimize.curve_fit(self.model_f, x_coordinates, y_coordinates)
                a_opt, b_opt, c_opt, d_opt = p_opt
                #GET LINE
                x_model = np.linspace(min(x_coordinates), max(x_coordinates), 20)
                y_model = self.model_f(x_model, a_opt, b_opt, c_opt,d_opt)
                
                #Fitting line on y axis.
                cordinates_list_model = list(zip(y_model, x_model))
                cordinates_list_model = [list(coord) for coord in cordinates_list_model]

                #Shift back to image frame from cropped image
                sampled_coordinates = [[round(x), round(y)] for x, y in cordinates_list_model]
                
                # Shift coordinates to center of the image
                shifted_coordinates = [(x - self.height//2, self.width//2 - y) for x, y in sampled_coordinates]

                # Convert shifted Cartesian coordinates to polar coordinates
                polar_coordinates = [[np.sqrt(x**2 + y**2), np.arctan2(y, x)]for x, y in shifted_coordinates]
                
                #return original image (x,y) and polar cords when at center of image
                return sampled_coordinates, polar_coordinates

            except RuntimeError:
                print("No solution found")
                return [],[]


    def vx_to_odom_frame_tf(self, polar_list):
        """
        Transfer it to odom frame.

        
        Args:
            polar_list: List of fitted points in polar system with origin being center of image

        Returns:
            cartesian_coordinates: List of fitted points in Odom frame
        """
        vx_theta = self.vx_yaw
        # Discout vx_yaw
        for points in polar_list:
                points[1] = self.roll_over_radians( points[1] - vx_theta)

        # Convert to Cartesian (x,y)
        cartesian_coordinates = [[round(r * np.cos(theta)), round(r * np.sin(theta))] for r, theta in polar_list]
        # Shift back to Image frame
        cartesian_coordinates = [[int(x+self.width//2),int(self.height//2 - y)] for x, y in cartesian_coordinates]
        # cartesian_coordinates = sorted(cartesian_coordinates, key=lambda coord: coord[1], reverse=True)

        return cartesian_coordinates
    
    def find_cordinates_of_max_value(self, image):
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
        #returns in [x,y]
        return max_intensity_coordinates
    
    def roll_over_radians(self, angle, range_start=-math.pi, range_end=math.pi):
        """
        Wrap angles over [-pi, pi]


        Args:
            angle: Angle in radians.

        Returns:
            rolled_angle: Angle in radians.
        """
        rolled_angle = (angle - range_start) % (range_end - range_start) + range_start
        return rolled_angle
    
    def get_usable_edges(self, coordinates):
        """
        Use it to find the outer edge from the canny image. 
        The resultant is the cordinates of max likelihood where the obstacle is.

        Args:
            coordinates: [[x,y]] of edges from Image.

        Returns:
            cartesian_coordinates: List of points that are on the outer edge from canny image.
            polar_coordinates: List of points [r, theta] from the center of the image. (Odom frame, vx as origin)
        """
        polar_dict = {}

        # Step 1: Shift coordinates to center of the image
        shifted_coordinates = [(x - self.height//2, self.width//2 - y) for x, y in coordinates]

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
        # Step 5: Shift back
        cartesian_coordinates = [[int(x+self.width//2),int(self.height//2 - y),] for x, y in cartesian_coordinates]
        # Helps in plotting.
        cartesian_coordinates = [[y,x] for x,y in cartesian_coordinates]
       # Step 6: sample //TO-DO; Time based sampling.
        # if len(cartesian_coordinates) > self.width//4:
        #     diff = len(cartesian_coordinates) - self.width//4
        #     for i in range(diff):
        #         cartesian_coordinates.pop(i)
        #
        # [x,y] [r,theta]
        return cartesian_coordinates, polar_coordinates
    
    def plot_circles_vx_frame(self, coordinates_list, radius=1, color=255):
        """
        Viz function with to show vehicle frame

        Args:
            coordinates_list: List of points in polar system in Odom frame.

        Returns:
            image: Image with points plotted in vehicle frame.
            cartesian_coordinates: list of points in vehicle frame
        """
        yawd_p = [list(t) for t in coordinates_list]

        vx_theta = self.vx_yaw
        for points in yawd_p:
            points[1] = self.roll_over_radians(points[1] - np.float64(vx_theta))

        # Convert to Cartesian.
        cartesian_coordinates = [[round(r * np.cos(theta)), round(r * np.sin(theta))] for r, theta in yawd_p]
        # Shift back to Image frame
        cartesian_coordinates = [[int(x+self.width//2),int(self.height//2 - y)] for x, y in cartesian_coordinates]
        # Plotting takes y,x
        cartesian_coordinates = [[y,x] for x,y in cartesian_coordinates]

        # Create an empty image (numpy array)
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        # Iterate through the list of coordinates and draw circles
        for coordinates in cartesian_coordinates:
            center = tuple(coordinates)
            cv2.circle(image, center, radius, color, 1)

        # image = cv2.line(image, (0, 3*self.width//4), (200, 3*self.width//4), 100, 1)
        # image = cv2.line(image, (100, 0), (100, 200), 100, 1)

        #Draw Vx frame
        # image = cv2.line(image, (self.width//2, self.height//2), (self.width//2, self.height//2-5), 200, 1)
        # image = cv2.line(image, (self.width//2, self.height//2), (self.width//2-5, self.height//2), 200, 1) 

        return image, cartesian_coordinates
    
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
        list1 = [(y,x) for x,y in list1]
        
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
        path = Path()
        
        path.header.frame_id = self.frame
        path.header.stamp =  self.time
        # print(shifted_coordinates)
        for cords in shifted_coordinates:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.frame
            pose_stamped.header.stamp = self.time

            #IMAGE TO MAP FRAME FIXED TO ODOMs
            x = -((cords[1] - self.height//2) * self.resolution) + self.vx_x
            y = ((self.width//2 - cords[0])* self.resolution ) + self.vx_y

            pose_stamped.pose.position.x = x#self.height//4 - cords[1] * self.resolution
            pose_stamped.pose.position.y = y#self.height//4 - cords[0] * self.resolution 
            pose_stamped.pose.orientation.x = 0
            pose_stamped.pose.orientation.y = 0
            pose_stamped.pose.orientation.z = 0
            pose_stamped.pose.orientation.w = 1
            path.poses.append(pose_stamped)

        self.pub.publish(path)
    
if __name__ == "__main__":
    rospy.init_node('path_node')
    l = CostMapProcess()
    rospy.spin()