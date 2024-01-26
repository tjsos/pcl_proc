#!/usr/bin/env python3
import rospy
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np
import cv2
import math
import scipy

class CostMapProcess:
    def __init__(self) -> None:
        rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.mapCB)
        rospy.Subscriber("/alpha_rise/odometry/filtered/local", Odometry, self.odomCB)

        self.pub = rospy.Publisher("/stand_off_path", Path, queue_size=1)

        self.vx_y, self.vx_x, self.yaw = 0,0,0

    def odomCB(self, message):
        #Needed to update the path in odom_frame
        self.vx_x = message.pose.pose.position.x
        self.vx_y = message.pose.pose.position.y

        x = message.pose.pose.orientation.x
        y = message.pose.pose.orientation.y
        z = message.pose.pose.orientation.z
        w = message.pose.pose.orientation.w

        self.yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    def mapCB(self, message):
        """
        Costmap
        """
        self.width = message.info.width
        self.height = message.info.height
        data = message.data
        self.frame = message.header.frame_id
        self.time = message.header.stamp

        self.resolution = message.info.resolution
        
        """
        Image
        """
        # Costmap to Image
        data = np.array(data).reshape(self.width,self.height)
        data = data.astype(np.uint8)

        #Corrected for Costmap FRAME -> IMage frame.
        data = cv2.flip(data, 1)  
        data = cv2.rotate(data, cv2.ROTATE_90_CLOCKWISE)

        #Dilate Raw Measurements to get solid reading       
        dilate = cv2.dilate(data, (5,5), 2)
        dilate = cv2.dilate(dilate, (5,5), 2)
        dilate = cv2.dilate(dilate, (5,5), 2)
        dilate = cv2.dilate(dilate, (5,5), 2)
        dilate = cv2.dilate(dilate, (5,5), 2)


        """
        Edges
        """
        #Get Edge
        canny_image = cv2.Canny(dilate,200,255)
        #Find cordinates of the edges
        raw_pixels = self.find_cordinates_max_value(canny_image)
        #Get usable edges
        edge, edge_polar = self.get_edges(raw_pixels)
        #Edges in vx_frame
        vx_frame_edges = self.odom_to_vx_frame_tf(edge_polar)

        #//To-Do : Fit line. Upsample?
        #plot vx frame
        vx_frame_image = self.plot_circles_vx_frame(vx_frame_edges)
        
        path_cells, path_cells_polar = self.curve_fit(vx_frame_image)
        odom_frame_path = self.vx_to_odom_frame_tf(path_cells_polar)
        # print(odom_frame_path)
        """
        Path
        """
        self.pub_path(odom_frame_path)


        """
        Visualization
        """
        ## plot in odom frame
        # edge_image = self.plot_circles(edge)
        ## Compare the Canny and Custom
        compare = self.compare_two_lists(raw_pixels, odom_frame_path)

        mix= np.hstack((data, canny_image, compare, vx_frame_image))
        cv2.imshow("window", mix)
        cv2.waitKey(1)
    
    def model_f(self,x, a, b, c, d):
        # return a*(x-b)**2 + c
        return a*x**3 + b* x**2 +c*x +d
        
    def curve_fit(self, vx_frame):
        #Only interested in path frame
        vx_frame_cropped = vx_frame[ :3*self.width//4, self.height//2:]
        img_cropped_cart = self.find_cordinates_max_value(vx_frame_cropped)
        if img_cropped_cart is not []:
            x_coordinates, y_coordinates = zip(*img_cropped_cart)
            try:
                #FIT THE CURVE
                p_opt, p_cov = scipy.optimize.curve_fit(self.model_f, x_coordinates, y_coordinates)
                a_opt, b_opt, c_opt, d_opt = p_opt
                #GET LINE
                x_model = np.linspace(min(x_coordinates), max(x_coordinates), 20)
                y_model = self.model_f(x_model, a_opt, b_opt, c_opt,d_opt)
                #Y here is X in Image frame
                cordinates_list_model = list(zip(y_model, x_model))
                cordinates_list_model = [list(coord) for coord in cordinates_list_model]

                #Shift back to image frame from cropped image
                sampled_coordinates = [[round(x)+ self.height//2, round(y)] for x, y in cordinates_list_model]
                
                # Shift coordinates to center of the image
                shifted_coordinates = [[x - self.height//2, self.width//2 - y] for x, y in sampled_coordinates]

                # Convert shifted Cartesian coordinates to polar coordinates
                polar_coordinates = [[np.sqrt(x**2 + y**2), np.arctan2(x, y)]for x, y in shifted_coordinates]
                
                return sampled_coordinates, polar_coordinates

            except RuntimeError:
                print("No solution found")
                return [],[]


    def vx_to_odom_frame_tf(self, polar_list):
        """
        Transfer it to odom frame
        """
        vx_theta = self.yaw
        for points in polar_list:
             
            if round(vx_theta) <= 0:
                points[1] = self.roll_over_radians( -points[1] + vx_theta)
            elif round(vx_theta) > 0:
                points[1] = self.roll_over_radians( points[1] - vx_theta)

        # print(points[1])
        cartesian_coordinates = [[abs(round(r * np.sin(theta))), round(r * np.cos(theta))] for r, theta in polar_list]
        cartesian_coordinates = [[int(self.height//2 + x),int(self.width//2 - y),] for x, y in cartesian_coordinates]
        # cartesian_coordinates = sorted(cartesian_coordinates, key=lambda coord: coord[1], reverse=True)

        return cartesian_coordinates

    def odom_to_vx_frame_tf(self, polar_list):
        """
        Transfer it to vehicle frame
        """
        yawd_p = [list(t) for t in polar_list]

        vx_theta = self.yaw
        for points in yawd_p:
            points[1] = self.roll_over_radians(points[1] + np.float64(vx_theta))

       # Step 4: Convert to Cartesian.
        cartesian_coordinates = [[round(r * np.sin(theta)), round(r * np.cos(theta))] for r, theta in yawd_p]
        # Step 5: Shift back
        cartesian_coordinates = [[int(self.width//2 - x),int(y+self.height//2),] for x, y in cartesian_coordinates]
        return cartesian_coordinates
    
    def find_cordinates_max_value(self, image):
        """
        Find cordinates of white pixels
        """
        max_intensity = np.max(image)

        # Get the coordinates of pixels with the maximum intensity
        max_intensity_coordinates = np.column_stack(np.where(image == max_intensity))
        #returns in [x,y]
        return max_intensity_coordinates
    
    def roll_over_radians(self, angle, range_start=-math.pi, range_end=math.pi):
        """
        Wrap angles over [-pi, pi]
        """
        rolled_angle = (angle - range_start) % (range_end - range_start) + range_start
        return rolled_angle
    
    def get_edges(self, coordinates):
        """
        Use it to find the outer edge from the canny image. The resultant is the cordinates
        max likelihood where the obstacle is.
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
        # Step 4: Convert to Cartesian.
        cartesian_coordinates = [[round(r * np.sin(theta)), round(r * np.cos(theta))] for r, theta in polar_coordinates]
        # Step 5: Shift back
        cartesian_coordinates = [[int(self.width//2 - x),int(y+self.height//2),] for x, y in cartesian_coordinates]
        
       # Step 6: sample //TO-DO; Time based sampling.
        # if len(cartesian_coordinates) > self.width//4:
        #     diff = len(cartesian_coordinates) - self.width//4
        #     for i in range(diff):
        #         cartesian_coordinates.pop(i)
        #
        # [x,y] [r,theta]
        return cartesian_coordinates, polar_coordinates

    def plot_circles(self, coordinates_list, radius=1, color=155):
        """
        Viz function
        """
        # Create an empty image (numpy array)
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        # Iterate through the list of coordinates and draw circles
        for coordinates in coordinates_list:
            center = tuple(coordinates)
            cv2.circle(image, center, radius, color, 1)

        return image
    
    def plot_circles_vx_frame(self, coordinates_list, radius=1, color=155):
        """
        Viz function with added vx axis
        """
        # Create an empty image (numpy array)
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        # Iterate through the list of coordinates and draw circles
        for coordinates in coordinates_list:
            center = tuple(coordinates)
            cv2.circle(image, center, radius, color, 1)

        image = cv2.line(image, (0, 3*self.width//4), (200, 3*self.width//4), 100, 1)
        image = cv2.line(image, (100, 0), (100, 200), 100, 1)

        #Draw Vx frame
        # image = cv2.line(image, (self.width//2, self.height//2), (self.width//2, self.height//2-5), 200, 1)
        # image = cv2.line(image, (self.width//2, self.height//2), (self.width//2-5, self.height//2), 200, 1) 

        return image
    
    def compare_two_lists(self, list1, list2):
        """
        Viz function to plot the original canny and extracted edge
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