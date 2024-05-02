#!/usr/bin/env python3

#Author: Tony Jacob
#Part of RISE Project. 
#tony.jacob@uri.edu

import rospy
import math
import cv2
import csv
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Plot_StandOff:
    def __init__(self) -> None:
        rospy.Subscriber("/vx_image", Image, self.image_cB)
        self.bridge = CvBridge()
        self.prev_distance = 0
        self.start = time.time()
        self.f = open('/home/soslab/auv_ws/test4.csv', 'w')
        self.writer = csv.writer(self.f)

    def image_cB(self, msg):
        elapsed_time = time.time() - self.start
        if elapsed_time > 1:
            self.start = time.time()
            image = self.bridge.imgmsg_to_cv2(msg)
            height, widhth = image.shape
            distance = self.perpendicular_distance_to_first_non_zero_pixel(height, widhth, image)
            if distance != None:
                self.prev_distance = distance
            print(self.prev_distance * 0.5)
            self.writer.writerow([self.prev_distance * 0.5])
            cv2.imshow("image", image)
            cv2.waitKey(1)

    
    def perpendicular_distance_to_first_non_zero_pixel(self, image_height, image_width, image):
        # Calculate the center coordinates
        center_x = image_width // 2
        center_y = image_height // 2

        # Iterate horizontally from center towards right
        for x in range(center_x, image_width):
            # Check if the pixel is non-zero
            if image[center_y, x] != 0:
                # Calculate perpendicular distance to the pixel
                perpendicular_distance = x - center_x
                return perpendicular_distance

        # If no non-zero pixel found, return None
        return None



if __name__ == "__main__":
    rospy.init_node("plot_node")
    l = Plot_StandOff()
    rospy.spin()