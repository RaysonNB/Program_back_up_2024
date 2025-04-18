#!/usr/bin/env python3
import cv2
import numpy as np
from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge


def callBack_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def detect_color(frame):
    _frame = cv2.resize(frame, (40, 30))
    hsv_frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2HSV)
    clist = []
    global color
    for c in color:
        mask = cv2.inRange(hsv_frame, np.array(c[1]), np.array(c[2]))
        result = cv2.bitwise_and(_frame, _frame, mask=mask)
        clist.append([count_color(result), c[0]])
    return sorted(clist, reverse=True)[0][1]


def count_color(frame):
    h, w, c = frame.shape
    cnt = 0
    for x in range(w):
        for y in range(h):
            if frame[y, x, 0] != 0 and frame[y, x, 1] != 0 and frame[y, x, 2] != 0:
                cnt += 1
    return cnt


if __name__ == "__main__":
    rospy.init_node("color_detection")
    rospy.Subscriber("/cam2/color/image_raw", Image, callBack_image)

    color = [
        ["red", [175, 43, 46], [180, 255, 255]],
        ["orange", [0, 140, 100], [20, 255, 255]],
        ["yellow", [22, 93, 0], [33, 255, 255]],
        ["green", [34, 20, 0], [94, 255, 255]],
        ["blue", [94, 40, 2], [126, 255, 255]],
        ["purple", [130, 43, 46], [145, 255, 255]],
        ["pink", [125, 100, 30], [165, 255, 255]],
        ["white", [0, 0, 10], [25, 30, 255]],
        ["black", [22, 0, 0], [180, 255, 70]],
        ["brown", [20, 20, 0], [33, 140, 170]]
    ]

    _image = None

    while not rospy.is_shutdown():
        if _image is None:
            continue

        frame = _image.copy()

        # Detect color
        detected_color = detect_color(frame)
        cv2.putText(frame, f"Color: {detected_color}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Color Detection", frame)
        key = cv2.waitKey(1)
        if key in [27, ord('q')]:
            break

    cv2.destroyAllWindows()
    rospy.loginfo("Color detection ended")
