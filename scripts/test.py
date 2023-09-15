#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np

class Test:
    def __init__(self) -> None:
        rospy.Subscriber('/alpha_rise/msis/pointcloud', PointCloud2, self.pointcloud_callback)
        rospy.Subscriber("/alpha_rise/msis/stonefish/data/pointcloud/filtered", PointCloud2, self.pointcloud_callback2)

        
    def pointcloud_callback(self, msg):
        self.raw_data = msg.data

    def pointcloud_callback2(self, msg):
        self.filtered_data = msg.data
        self.compare(self.raw_data, self.filtered_data)
    
    def compare(self, raw, filtered):
        for i in range(len(raw)):
            diff = raw[i] - filtered[i]
        
        diff = np.array(diff)
        print(diff.min())

if __name__ == "__main__":
    rospy.init_node("test")
    t = Test()
    rospy.spin()
        