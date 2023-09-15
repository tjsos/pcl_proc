#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from math import nan

class Processing:
    def __init__(self) -> None:
        # Subscribe to the PointCloud2 topic
        # rospy.Subscriber('/alpha_rise/msis/stonefish/data/pointcloud', PointCloud2, pointcloud_callback)
        rospy.Subscriber('/alpha_rise/msis/pointcloud', PointCloud2, self.pointcloud_callback)
        
    def pointcloud_callback(self, pointcloud_msg):
        pcl_pub = rospy.Publisher("/alpha_rise/msis/stonefish/data/pointcloud/filtered", PointCloud2, queue_size=1)
        pcl_msg = PointCloud2()
        pcl_msg.header.frame_id = pointcloud_msg.header.frame_id
        pcl_msg.header.stamp = pointcloud_msg.header.stamp
        
        pcl_msg.fields = pointcloud_msg.fields
        pcl_msg.height = pointcloud_msg.height
        pcl_msg.width = pointcloud_msg.width
        pcl_msg.point_step = pointcloud_msg.point_step
        pcl_msg.row_step = pointcloud_msg.row_step
        pcl_msg.is_dense = pointcloud_msg.is_dense


        mean, std_dev, intensities = self.get_intensities(pointcloud_msg=pointcloud_msg)
    
        # Populate filtered pointclouds.
        points = np.zeros((pcl_msg.width,len(pcl_msg.fields)),dtype=np.float32)
        for index,point in enumerate(pc2.read_points(pointcloud_msg, skip_nans=True)):
            if index >= 120:
                ##Total bins = 1200. Set range = 20m. 1m = 60bins.
                x, y, z, i = point[:4]
                #Filter
                if i < mean+1.5 *std_dev:
                    i = 0               
                points[index][0] = x
                points[index][1] = y
                points[index][3] = i
            else:
                points[index][0] = nan
                points[index][1] = nan
                points[index][3] = nan
        pcl_msg.data = points.tobytes()
        pcl_pub.publish(pcl_msg)

        # print(point)
        # print("Point: x =", x, "y =", y, "z =", z, "i =", i)

        # if not -10<x<10:
        #     if not -10<y<10:
                # x, y, z, i = point[:4]

        #         print("Point: x =", x, "y =", y, "z =", z, "i =", i)
        #         pcl_pub.publish(pcl_msg)
    
    def get_intensities(self,pointcloud_msg):
        """
        Returns the mean, std_dev and the echo intensity arrays.
        """
        intensities = []
        for index,point in enumerate(pc2.read_points(pointcloud_msg, skip_nans=True)):
            x, y, z, i = point[:4]
            intensities.append(i)
        intensities = np.array(intensities)
        mean = np.mean(intensities)
        std_dev = np.std(intensities)
        return mean, std_dev, intensities.tolist()
    
if __name__ == '__main__':
    rospy.init_node('pointcloud_subscriber', anonymous=True)
    process=Processing()
    rospy.spin()