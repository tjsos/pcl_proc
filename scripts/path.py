#!/usr/bin/env python3
import rospy
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np
import cv2


class CostMapProcess:
    def __init__(self) -> None:
        self.sub = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.mapCB)
        self.pub = rospy.Publisher("/stand_off_path", Path, queue_size=1)
        self.pub_filtered = rospy.Publisher("/stand_off_path/filtered", Path, queue_size=1)
        rospy.Subscriber("/alpha_rise/odometry/filtered/local", Odometry, self.odomCB)

        self.vx_y, self.vx_x = 0,0

    def odomCB(self, message):
        self.vx_x = message.pose.pose.position.x
        self.vx_y = message.pose.pose.position.y

    def mapCB(self, message):
        self.width = message.info.width
        self.height = message.info.height
        data = message.data
        self.frame = message.header.frame_id
        self.time = message.header.stamp

        self.resolution = message.info.resolution
        
        x = message.info.origin.orientation.x
        y = message.info.origin.orientation.y
        z = message.info.origin.orientation.z
        w = message.info.origin.orientation.w

        norm =  np.linalg.norm([x,y,z,w])
        q_normalised = [x,y,z,w]/norm
        x,y,z,w =  q_normalised
        yaw= np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        self.yaw = yaw

        # Costmap to Image
        data = np.array(data).reshape(self.width,self.height)
        data = data.astype(np.uint8)

        #Corrected for Costmap FRAME -> IMage frame.
        data = cv2.flip(data, 1)  
        data = cv2.rotate(data, cv2.ROTATE_90_CLOCKWISE)

        # near_img = cv2.resize(data,None, fx = 1, fy = 1, interpolation = cv2.INTER_NEAREST)
        
        dilate = cv2.dilate(data, (5,5), 2)
        dilate = cv2.dilate(dilate, (5,5), 2)
        dilate = cv2.dilate(dilate, (5,5), 2)
        dilate = cv2.dilate(dilate, (5,5), 2)
        dilate = cv2.dilate(dilate, (5,5), 2)

        canny = cv2.Canny(dilate,200,255)

        cell_row = self.find_highest_intensity_coordinates_per_row(canny)

        # row_points, col_points = self.find_closest_points_to_middle(canny)

        # print(row_points)

        # # erode = cv2.erode(dilate, (5,5), 1)
        
        # cell_row = self.find_highest_intensity_coordinates_per_row(data)
        
    #     #IMAGES
        edge = self.plot_circles(cell_row)
    #     edge_shifted = self.stand_off(cell_row, edge)
        
    #     #PATH NOT FILTERD
        path = self.pub_path(cell_row)
        self.pub.publish(path)

    #     #FILERD
    #     cell_row = self.find_highest_intensity_coordinates_per_custom_row(data, 2)
    #    # cell_row = self.remove_jumps(cell_row)

    #     path = self.pub_path(cell_row[5:-5])
        # self.pub_filtered.publish(path)

        # print(cell_row)
        mix= np.hstack((data, dilate, canny, edge))

        cv2.imshow("window", mix)
        cv2.waitKey(1)


    
    def find_highest_intensity_coordinates_per_row(self, image):
        #Gets the edge.
        # Read the image

        # Get the number of rows and columns in the image
        rows, cols = image.shape

        # Initialize a list to store coordinates of the last non-zero pixel for each row
        last_nonzero_coordinates_list = []

        # Iterate through rows
        for row_index in range(rows):
            # Find the indices of non-zero elements in the current row
            nonzero_indices = np.nonzero(image[row_index, :])

            # Check if there are any non-zero elements in the row
            if len(nonzero_indices[0]) > 0:
                # Get the coordinates of the last non-zero element in the row
                last_nonzero_col = nonzero_indices[0][0]

                # Append the coordinates to the list
                last_nonzero_coordinates_list.append((last_nonzero_col,row_index))

        return last_nonzero_coordinates_list
    
    # def find_highest_intensity_coordinates_per_custom_row(self, image, sample_rate):
    #         #Gets the edge.
    #         # Read the image

    #         # Get the number of rows and columns in the image
    #         rows, cols = image.shape

    #         # Initialize a list to store coordinates of the last non-zero pixel for each row
    #         last_nonzero_coordinates_list = []

    #         # Iterate through rows
    #         for row_index in range(0,rows,sample_rate):
    #             # Find the indices of non-zero elements in the current row
    #             nonzero_indices = np.nonzero(image[row_index, :])

    #             # Check if there are any non-zero elements in the row
    #             if len(nonzero_indices[0]) > 0:
    #                 # Get the coordinates of the last non-zero element in the row
    #                 last_nonzero_col = nonzero_indices[0][0]

    #                 # Append the coordinates to the list
    #                 last_nonzero_coordinates_list.append((last_nonzero_col,row_index))

    #         return last_nonzero_coordinates_list

    def plot_circles(self, coordinates_list, radius=1, color=155):
        # Plot just the edge.
        # Create an empty image (numpy array)
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        # Iterate through the list of coordinates and draw circles
        for coordinates in coordinates_list:
            center = tuple(coordinates)
            cv2.circle(image, center, radius, color, 1)

        return image
    
    # def stand_off(self, coordinates_list, image, radius = 1, color= 255):
    #     # PLot the standoff.
        
    #     for coordinates in coordinates_list:
    #         shifted_coordinates = (coordinates[0] - 10, coordinates[1])
    #         center = tuple(shifted_coordinates)
    #         cv2.circle(image, center, radius, color, 1)

    #     return image
    
    def pub_path(self, shifted_coordinates):
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
    
    # def remove_jumps(self, coordinates, window_size=10, threshold=1):
    #     """
    #     Remove sudden jumps from a list of coordinates using a moving average filter.

    #     Parameters:
    #     - coordinates: List of tuples (x, y).
    #     - window_size: Size of the moving average window.
    #     - threshold: Maximum allowable difference between the original and smoothed values.

    #     Returns:
    #     - List of filtered coordinates.
    #     """
    #     x_values, y_values = zip(*coordinates)

    #     # Apply a moving average filter to smooth the data
    #     # smoothed_y = np.convolve(y_values, np.ones(window_size) / window_size, mode='same')
    #     smoothed_x = np.convolve(x_values, np.ones(window_size) / window_size, mode='same')


    #     # Identify jumps based on the threshold
    #     # jump_mask_y = np.abs(y_values - smoothed_y) > threshold
    #     jump_mask_x = np.abs(x_values - smoothed_x) > threshold


    #     # Replace jumped values with smoothed values
    #     # filtered_y = np.where(jump_mask_y, smoothed_y, y_values)
    #     filtered_x = np.where(jump_mask_x, smoothed_x, x_values)


    #     # Create a list of filtered coordinates
    #     filtered_coordinates = list(zip(filtered_x, y_values))
        

    #     return filtered_coordinates
    
    
if __name__ == "__main__":
    rospy.init_node('path_node')
    l = CostMapProcess()
    rospy.spin()