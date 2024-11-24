#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
from pcms.openvino_models import HumanPoseEstimation
# astrapro
def callback_image1(msg):
    global _image1
    _image1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth1(msg):
    global _depth1
    _depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")


# gemini2
def callback_image2(msg):
    global _frame2
    _frame2 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth2(msg):
    global _depth2
    _depth2 = CvBridge().imgmsg_to_cv2(msg, "passthrough")

if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")

    # main
    print("astra rgb")
    _image1 = None
    _topic_image1 = "/camera/color/image_raw"
    rospy.Subscriber(_topic_image1, Image, callback_image1)
    rospy.wait_for_message(_topic_image1,Image)
    print("astra depth")
    _depth1 = None
    _topic_depth1 = "/camera/depth/image_raw"
    rospy.Subscriber(_topic_image1, Image, callback_depth1)
    # _sub_up_cam_image.unregister()
    #rospy.wait_for_message(_topic_depth1,Image)
    #
    print("gemini2 rgb")
    _frame2 = None


    print("gemini2 depth")
    _depth2 = None

    dnn_human_pose = HumanPoseEstimation("/home/pcms/models/openvino")
    print("finish load")
    while not rospy.is_shutdown():
        # voice check
        # break
        rospy.Rate(10).sleep()
        #if _frame2 is None: print("down rgb none")
        #if _depth2 is None: print("down depth none")
        if _depth1 is None: print("up depth none")
        if _image1 is None: print("up rgb none")

        if _depth1 is None or _image1 is None or _depth2 is None or _frame2 is None: continue
        
        down_image = _frame2.copy()
        down_depth = _depth2.copy()
        up_image = _image1.copy()
        up_depth = _depth1.copy()
        '''
        h, w, c = up_image.shape
        upout = cv2.line(up_image, (320, 0), (320, 500), (0, 255, 0), 5)
        downout = cv2.line(down_image, (320, 0), (320, 500), (0, 255, 0), 5)
        img = np.zeros((h, w * 2, c), dtype=np.uint8)
        img[:h, :w, :c] = upout
        img[:h, w:, :c] = downout
        '''

        poses = dnn_human_pose.forward(_image1)
        frame = dnn_human_pose.draw_poses(frame, poses, 0.1)
        for pose in poses:
            for i, p in enumerate(pose):
                x, y, c = map(int, p)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX,  1, (255,0,0), 1, cv2.LINE_AA) 
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
