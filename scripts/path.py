#!/usr/bin/env python3
import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class CostMapProcess:
    def __init__(self) -> None:
        self.sub = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.mapCB)

    def mapCB(self, message):
        width = message.info.width
        height = message.info.height
        data = message.data

        data = np.array(data).reshape(400,400)
        data = data.astype(np.uint8)
        for i in range(width):
            for j in range(height):
                if data[i][j] > 0:
                    data[i][j] = 255
                # else:
                #     data[i][j] = 0

        data = cv2.flip(data, 1)
        data = cv2.rotate(data, cv2.ROTATE_90_CLOCKWISE)


        near_img = cv2.resize(data,None, fx = 1, fy = 1, interpolation = cv2.INTER_NEAREST)

        # dilate = cv2.dilate(data, (5,5), 2)
        # erode = cv2.erode(dilate, (5,5), 1)
             
        mix= np.hstack((data, near_img))
        cv2.imshow("window", mix)
        cv2.waitKey(1)



if __name__ == "__main__":
    rospy.init_node('path_node')
    CostMapProcess()
    rospy.spin()