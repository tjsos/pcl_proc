#!/usr/bin/env python3
import rospy
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np
import cv2
import math

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
        # print(self.yaw)

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
        canny = cv2.Canny(dilate,200,255)
        #Find cordinates of the edges
        pixels = self.find_cordinates_max_value(canny)
        #Get usable edges
        cell_row, cell_row_polar = self.get_edges(pixels)
        #Edges in vx_frame
        vx_frame_edges = self.vx_frame_tf(cell_row_polar)

        #//To-Do : Fit line. Upsample?
        """
        Path
        """
        path = self.pub_path(cell_row)
        self.pub.publish(path)


        """
        Visualization
        """
        #plot in odom frame
        edge = self.plot_circles(cell_row)
        #plot vx frame
        vx_frame = self.plot_circles_vx_frame(vx_frame_edges)
        #Compare the Canny and Custom
        compare = self.compare_canny_edge(pixels, cell_row)

        mix= np.hstack((data, canny, compare, vx_frame))
        cv2.imshow("window", mix)
        cv2.waitKey(1)

    def vx_frame_tf(self, polar_list):
        """
        Transfer it to vehicle frame
        """
        yawd_p = [list(t) for t in polar_list]

        theta = self.yaw
        for points in yawd_p:
            points[1] = self.roll_over_radians(points[1] + np.float64(theta))

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
        image = cv2.line(image, (self.width//2, self.height//2), (self.width//2, self.height//2-10), 200, 1)
        image = cv2.line(image, (self.width//2, self.height//2), (self.width//2-10, self.height//2), 200, 1) 

        return image
    
    def compare_canny_edge(self, list1, list2):
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
        Create the path message
        """
        path = Path()
        
        path.header.frame_id = self.frame
        path.header.stamp =  self.time
        # print(shifted_coordinates)
        for cords in shifted_coordinates:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.frame
            pose_stamped.header.stamp = self.time

            #IMAGE TO MAP FRAME
            x = -((cords[1] - self.height//2) * self.resolution) + self.vx_x
            y = ((self.width//2 - cords[0])* self.resolution ) + self.vx_y

            pose_stamped.pose.position.x = x#self.height//4 - cords[1] * self.resolution
            pose_stamped.pose.position.y = y#self.height//4 - cords[0] * self.resolution 
            pose_stamped.pose.orientation.x = 0
            pose_stamped.pose.orientation.y = 0
            pose_stamped.pose.orientation.z = 0
            pose_stamped.pose.orientation.w = 1
            path.poses.append(pose_stamped)

        return path
    
    
if __name__ == "__main__":
    rospy.init_node('path_node')
    l = CostMapProcess()
    rospy.spin()