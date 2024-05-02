#!/usr/bin/env python3

#Author: Tony Jacob
#Part of RISE Project. 
#tony.jacob@uri.edu

import rospy
import math
import cv2
import csv
import time
from nav_msgs.msg import Odometry

class Plot_Odom:
    def __init__(self) -> None:
        rospy.Subscriber("/alpha_rise/odometry/filtered/local", Odometry, self.image_cB)
        self.start = time.time()
        self.f = open('/home/soslab/auv_ws/test5_odom.csv', 'w')
        self.writer = csv.writer(self.f)

    def image_cB(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.writer.writerow([x,y])
        print(x,y)
if __name__ == "__main__":
    rospy.init_node("plot_odom_node")
    l = Plot_Odom()
    rospy.spin()