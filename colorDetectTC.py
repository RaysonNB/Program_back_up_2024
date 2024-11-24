#!/usr/bin/env python3
import cv2
import numpy as np
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy

def callBack_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg,"bgr8")

def count_color(frame):
    h,w,c = frame.shape
    c=0
    for x in range(w):
        for y in range(h):
            if frame[y,x,0] != 0 and frame[y,x,1] != 0 and frame[y,x,1] != 0:
                c+=1
    return c

def detect_color(frame):
    _frame = cv2.resize(frame, (40, 30))
    hsv_frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2HSV)
    clist = []
    color = [["red", [165, 43, 46], [180, 255, 255]], ["orange", [0, 25, 0], [20, 255, 255]], ["yellow", [22, 93, 0], [33, 255, 255]],
             ["green", [34, 0, 0], [94, 255, 255]], ["blue", [94, 10, 2], [126, 255, 255]], ["purple", [130, 43, 46], [145, 255, 255]],
             ["pink", [125, 100, 30], [156, 255, 255]], ["white", [0, 0, 0], [126, 50, 255]], ["black", [0, 0, 0], [180, 255, 46]],
             ["gray", [0, 0, 64], [0, 0, 229]]]
    for c in color:
        mask = cv2.inRange(hsv_frame, np.array(c[1]), np.array(c[2]))
        result = cv2.bitwise_and(_frame, _frame, mask=mask)
        clist.append([count_color(result), c[0]])
    print(sorted(clist,reverse=True))
    return sorted(clist,reverse=True)[0][1]

def color_filter(image,hsv_frame,low,high):
    return 0
def detect_colorV2(frame):
    hsv_frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2HSV)
    color_list = ["red","orange","yellow","green","blue","purple","black","white","gray"]
    red = color_filter(frame,hsv_frame,[120,254,200],[179,255,255])

    red = color_filter(frame,hsv_frame,[5,75,0],[21,255,255])

    red = color_filter(frame,hsv_frame,[22,93,0],[179,255,255])

    red = color_filter(frame,hsv_frame,[120,254,200],[179,255,255])
if __name__ == "__main__":
    rospy.init_node("colorDetectTC")
    rospy.loginfo("program start")

    _image = None
    rospy.Subscriber("/cam2/color/image_raw", Image, callBack_image)
    #rospy.wait_for_message("/cam2/color/image_raw",Image)
    rospy.loginfo("camera finish")

    color = [["red", [175, 43, 46], [180, 255, 255]], ["orange", [0, 140, 100],[20, 255, 255]], ["yellow", [22, 93, 0], [33, 255, 255]],
                ["green", [34, 20, 0], [94, 255, 255]], ["blue", [94, 40, 2], [126, 255, 255]], ["purple", [130, 43, 46], [145, 255, 255]],
                ["pink", [125, 100, 30], [165, 255, 255]], ["white", [0, 0, 10], [25, 30, 255]], ["black", [22, 0, 0], [180, 255, 70]],
                ["gray", [0, 0, 64], [0, 0, 229]],["brown",[20, 20, 0],[33, 140, 170]]]
    #h 角度，v 濃度，s 深淺
    while not rospy.is_shutdown():

        rospy.Rate(20).sleep()
        if _image is None: continue
        #print("yes")
        frame = _image.copy()
        #print(frame.shape)
        _frame = cv2.resize(frame, (320, 240))
        #_frame = frame.copy()
        hsv_frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2HSV)
        for c in color:
            mask = cv2.inRange(hsv_frame, np.array(c[1]), np.array(c[2]))
            result = cv2.bitwise_and(_frame, _frame, mask=mask)
            cv2.imshow(c[0], result)

        #cv2.imshow("_frame",_frame)
        key = cv2.waitKey(1)
        if key == 27 or key == 32:
            break
        time.sleep(0.1)


