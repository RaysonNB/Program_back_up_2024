#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import yaml, time, cv2
from cv_bridge import CvBridge
from pcms.openvino_models import Yolov8

def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")
    
if __name__ == "__main__":
    rospy.init_node("test_chair")
    rospy.loginfo("task started!")

    _frame = None
    rospy.Subscriber("/cam2/color/image_raw", Image, callback_image)
    print("model loading")
    dnn_yolo = Yolov8("chair_eindhoven",device_name="GPU")
    dnn_yolo.classes = ["chair"]
    print("ready")
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        if _frame is None: continue
        image = _frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = dnn_yolo.forward(image)[0]["det"]
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, score, class_id = map(int, detection)
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 5)
        cv2.imshow("frame", image)
        cv2.waitKey(1)
    rospy.loginfo("end")
        
