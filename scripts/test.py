#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import math
import cv2
import tf

class Test:
    def __init__(self) -> None:
        # rospy.Subscriber('/alpha_rise/msis/pointcloud', PointCloud2, self.pointcloud_callback)
        self.x, self.y = 0.0,0.0
        rospy.Subscriber("/alpha_rise/msis/stonefish/data/pointcloud/filtered", PointCloud2, self.pointcloud_callback2)
        # rospy.Subscriber("/alpha_rise/msis/pointcloud", PointCloud2, self.pointcloud_callback2)
        
        rospy.Subscriber("/alpha_rise/odometry/filtered/local", Odometry, self.odomCB)
        self.map_publisher = rospy.Publisher('/my_occupancy_grid', OccupancyGrid, queue_size=1)
        self.width, self.height = 1300,1300
        self.listener = tf.TransformListener()
        self.image = np.zeros((self.width,self.height), dtype= np.int8)

    def quaternion_rotation_matrix(self,Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.
    
        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
    
        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]
        
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
                                
        return rot_matrix
    
    def quaternion_to_euler(self, w, x, y, z):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw).

        Args:
            w (float): Scalar (real) part of the quaternion.
            x (float): x component of the quaternion.
            y (float): y component of the quaternion.
            z (float): z component of the quaternion.

        Returns:
            tuple: A tuple containing roll, pitch, and yaw angles in radians.
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    
    def odomCB(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        self.x_rot = msg.pose.pose.orientation.x
        self.y_rot = msg.pose.pose.orientation.y
        self.z_rot = msg.pose.pose.orientation.z
        self.w_rot = msg.pose.pose.orientation.w

        self.roll, self.pitch, self.yaw = self.quaternion_to_euler(self.w_rot,self.x_rot, self.y_rot, self.z_rot)

        # (trans,rot) = self.listener.lookupTransform('alpha_rise/odom', 'alpha_rise/ping360_link', rospy.Time())
        # rot_matrix = self.quaternion_rotation_matrix(rot)
        # self.T_odom_to_lidar = np.array([[rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], trans[0]],
        #                     [rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], trans[1]],
        #                     [rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], trans[2]],
        #                     [0,                 0,                  0,                  1]])
        
    def project_point(self, x, y, theta):
        # Calculate the distance from the origin
        r = math.sqrt(x**2 + y**2)

        # Calculate the new angle relative to the positive x-axis
        theta_new = math.atan2(y, x) + theta

        # Calculate the new coordinates of the projected point
        x_proj = r * math.cos(theta_new)
        y_proj = r * math.sin(theta_new)

        return x_proj, y_proj
    
    def pointcloud_callback2(self, msg):
        # Create an instance of the OccupancyGrid message
        occupancy_grid = OccupancyGrid()

        # Set the necessary fields of the message
        occupancy_grid.header.frame_id = 'alpha_rise/odom' #'wamv/msis'  # Frame ID for the occupancy grid
        occupancy_grid.info.resolution = 0.01666667#1cell/m

        occupancy_grid.info.width = self.width #self.width  # Width of the grid in cells
        occupancy_grid.info.height =self.height #self.height  # Height of the grid in cells
        occupancy_grid.info.origin.position.x =  self.x - (occupancy_grid.info.width/2*occupancy_grid.info.resolution)
        occupancy_grid.info.origin.position.y =  self.y - (occupancy_grid.info.height/2*occupancy_grid.info.resolution)
        
        # occupancy_grid.info.origin.orientation.x = self.x_rot
        # occupancy_grid.info.origin.orientation.y = self.y_rot
        # occupancy_grid.info.origin.orientation.z = self.z_rot
        # occupancy_grid.info.origin.orientation.w = self.w_rot


        # for i in range(600):
        #     self.image[i][i] = 255

        #To;Do NEED TO READ PCL FROM ODOM FRAME
        for index,point in enumerate(pc2.read_points(msg, skip_nans=True)):
            x, y, z, i = point[:4]
            
            # print(round(x * 1/occupancy_grid.info.resolution),round(y * 1/occupancy_grid.info.resolution))
            # self.image[x][y] = i        
        #     # P_lidar = np.array([x, y, z, 1])
        #     # P_odom = np.dot(self.T_odom_to_lidar, P_lidar)
        #     # x = P_odom[0]
        #     # y = P_odom[1]
        #     # z = P_odom[2]
        #     # print(x, y, z)
            x, y = self.project_point(x,y,self.yaw)
            
        #     print(self.height/2-abs(round(x * 1 / occupancy_grid.info.resolution)), self.height/2-abs(round(y * 1 / occupancy_grid.info.resolution)))
            if x>= 0 and y >0:
                self.image[self.height//2+ abs(round(x * 1 / occupancy_grid.info.resolution))][self.height//2+abs(round(y * 1/ occupancy_grid.info.resolution))] = 255#i
            
            elif x<0 and y<0:
                self.image[self.height//2- abs(round(x * 1 / occupancy_grid.info.resolution))][self.height//2 - abs(round(y * 1/ occupancy_grid.info.resolution))] = 255#i
            
            elif x >=0 and y<0:
                self.image[self.height//2+ abs(round(x * 1 / occupancy_grid.info.resolution))][self.height//2-abs(round(y * 1/ occupancy_grid.info.resolution))] = 255#i
            
            elif x<0 and y>=0:
                self.image[self.height//2 - abs(round(x * 1 / occupancy_grid.info.resolution))][self.height//2+abs(round(y * 1/ occupancy_grid.info.resolution))] = 255#i
        
        
        # Populate the occupancy data (array of cell values)
        # Here, we assume the occupancy data is a 1D list of cell values
        
        image = cv2.rotate(self.image, cv2.ROTATE_180)

        image = cv2.flip(image, 0)
        # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        occupancy_data = image.flatten().astype(int).tolist()#image.flatten()
        occupancy_grid.data = occupancy_data #* occupancy_grid.info.height * occupancy_grid.info.height

        # # Publish the occupancy grid
  # Adjust the publishing rate as needed

        # occupancy_grid.header.stamp = rospy.Time.now()  # Update the timestamp
        self.map_publisher.publish(occupancy_grid)

    def get_intensities(self, msg):
        beam_intensity = []
        for index,point in enumerate(pc2.read_points(msg, skip_nans=True)):
            x, y, z, i = point[:4]
            # print(f'X:{x}, Y:{y}, I:{i}')
            beam_intensity.append([self.distance_and_angle(x, y), i])
        print(beam_intensity)


    def distance_and_angle(self, x, y):
        # Calculate distance using the Pythagorean theorem
        distance = math.sqrt(x**2 + y**2)

        # Calculate angle (in radians) using arctan2
        angle = math.atan2(y, x)

        # Convert angle to degrees
        angle_degrees = math.degrees(angle)
        if angle_degrees < 0:
            angle_degrees = 360 - angle_degrees

        return distance, 360-angle_degrees
    
if __name__ == "__main__":
    rospy.init_node("test")
    t = Test()
    rospy.spin()
        