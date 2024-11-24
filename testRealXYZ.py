#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pcms.openvino_models import HumanPoseEstimation, PersonAttributesRecognition
import math
def callBack_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg,"bgr8")

def callBack_depth(msg):
    global _depth
    _depth = CvBridge().imgmsg_to_cv2(msg,"passthrough")

def get_real_xyz(x, y, s):
    numy,numz=0,0
    x,y=int(x),int(y)
    global _depth,A,B
    if s=="down": numy,numz=-250,230
    else: numy,numz=-1050,80
    a = 50.0 * np.pi / 180
    b = 60.0 * np.pi / 180
    '''
    tx = x - 320
    tx = np.tan(43 * np.pi / 180) / np.tan(45.5 * np.pi / 180) * tx
    x = tx + 320
    x = int(x)
    ty = y - 240
    ty = np.tan(27.5 * np.pi / 180) / np.tan(33 * np.pi / 180) * ty
    y = ty + 240
    y = int(y)
    '''
    print(f"x = {x}. y = {y}")
    #print("depth shape",_depth.shape)
    #_depth = cv2.resize(_depth,(int(A),int(B)))
    d = _depth[y][x]
    print(f"d: {d}")
    h, w = _depth.shape[:2]
    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    #print(real_x, real_y)
    return real_x, real_y, d+numz

if __name__ == "__main__":
    rospy.init_node("task2")
    rospy.loginfo("program start")

    A=(640*math.tan((45.5*math.pi)/180))/math.tan((43*math.pi)/180)
    B =(480*math.tan((33*math.pi)/180))/math.tan((27.5*math.pi)/180)
    _image = None
    rospy.Subscriber("/cam2/color/image_raw", Image, callBack_image)
    #rospy.wait_for_message("/cam2/color/image_raw",Image)
    rospy.loginfo("camera finish")
    
    _depth = None
    rospy.Subscriber("/cam2/depth/image_raw", Image, callBack_depth)
    path_openvino = "/home/pcms/models/openvino"
    dnn_pose = HumanPoseEstimation(path_openvino)
    print("start")
    
    print(A)
    print(B)    
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        #print(math.tan((45.5*math.pi)/180))
        if _image is None: continue
        poses = dnn_pose.forward(_image)
        if len(poses) > 0:
            YN = -1
            a_num, b_num = 9, 7
            for pose in poses:
                yu = 0
                if pose[9][2] > 0 and pose[7][2] > 0:
                    print(f"realx={pose[9][0]},realy={pose[9][1]}")
                    #px9,py9=(A-640)/2+pose[9][0],(B-480)/2+pose[9][1]
                    #print(f"{px9=} {py9=}")
                    x9,y9,d=get_real_xyz(int(pose[9][0]),int(pose[9][1]),"up")
                    cv2.circle(_image,(int(pose[9][0]),int(pose[9][1])),3,(255,255,0),-1)
                    #cv2.circle(up_image, (A[0], A[1]), 3, (255, 255, 0), -1)
                    print(f"x9:{x9},y9:{y9},d7:{d}")    
                    #x7,y7,d=get_real_xyz(pose[7][0],pose[7][1],"up")
                    #print("\n")
                    #cv2.circle(frame,(x7,y7),3,(0,0,255),-1)
                    #print(f"x7:{x7},y7:{y7},d7:{d}")
        cv2.imshow("image",_image)
        #cv2.imshow("depth",_depth)
        #print(f"{_image.shape = }, {_depth.shape = }")
        key_code = cv2.waitKey(1)
        if key_code in [27,ord('q')]:
            break
rospy.loginfo("task2 end")
