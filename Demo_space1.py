#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import HumanPoseEstimation,FaceDetection
import numpy as np
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from RobotChassis import RobotChassis

from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
import subprocess
import math
def say(a):
    text = str(a)
    process = subprocess.Popen(['espeak-ng', '-v', 'yue', '-a', '300', '-s', '180', text])
def say2(a):
    text = str(a)
    process = subprocess.Popen(['espeak-ng', '-v', 'yue', '-a', '200', '-s', '200', text])
    process.wait()
# gemini2
def callback_image2(msg):
    global frame2
    frame2 = CvBridge().imgmsg_to_cv2(msg, "bgr8")
def callback_depth2(msg):
    global depth2
    depth2 = CvBridge().imgmsg_to_cv2(msg, "passthrough")
def callback_imu(msg):
    global _imu
    _imu = msg
def get_distance(px, py, pz, ax, ay, az, bx, by, bz):
    A, B, C, p1, p2, p3, qx, qy, qz, distance = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    A = int(bx) - int(ax)
    B = int(by) - int(ay)
    C = int(bz) - int(az)
    p1 = int(A) * int(px) + int(B) * int(py) + int(C) * int(pz)
    p2 = int(A) * int(ax) + int(B) * int(ay) + int(C) * int(az)
    p3 = int(A) * int(A) + int(B) * int(B) + int(C) * int(C)
    # print("1",p1,p2,p3)
    if (p1 - p2) != 0 and p3 != 0:
        t = (int(p1) - int(p2)) / int(p3)
        qx = int(A) * int(t) + int(ax)
        qy = int(B) * int(t) + int(ay)
        qz = int(C) * int(t) + int(az)
        return int(int(pow(((int(qx) - int(px)) ** 2 + (int(qy) - int(py)) ** 2 + (int(qz) - int(pz)) ** 2), 0.5)))
    return 0
def get_real_xyz(dp, x, y):
    a = 55.0 * np.pi / 180
    b = 86.0 * np.pi / 180
    d = dp[y][x]
    h, w = dp.shape[:2]
    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d
def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)
def turn_to(angle: float, speed: float):
    global _imu
    max_speed = 0.05
    limit_time = 5
    start_time = rospy.get_time()
    while True:
        q = [
            _imu.orientation.x,
            _imu.orientation.z,
            _imu.orientation.y,
            _imu.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(q)
        e = angle - yaw
        print(yaw, e)
        if yaw < 0 and angle > 0:
            cw = np.pi + yaw + np.pi - angle
            aw = -yaw + angle
            if cw < aw:
                e = -cw
        elif yaw > 0 and angle < 0:
            cw = yaw - angle
            aw = np.pi - yaw + np.pi + angle
            if aw < cw:
                e = aw
        if abs(e) < 0.01 or rospy.get_time() - start_time > limit_time:
            break
        move(0.0, max_speed * speed * e)
        rospy.Rate(20).sleep()
    move(0.0, 0.0)
def turn(angle: float):
    global _imu
    q = [
        _imu.orientation.x,
        _imu.orientation.y,
        _imu.orientation.z,
        _imu.orientation.w
    ]
    roll, pitch, yaw = euler_from_quaternion(q)
    target = yaw + angle
    if target > np.pi:
        target = target - np.pi * 2
    elif target < -np.pi:
        target = target + np.pi * 2
    turn_to(target, 0.1)
def calc_linear_x(cd, td):
    if cd <= 0: return 0
    e = cd - td
    p = 0.0005
    x = p * e
    if x > 0: x = min(x, 0.5)
    if x < 0: x = max(x, -0.5)
    return x
def calc_angular_z(cx, tx):
    if cx < 0: return 0
    e = tx - cx
    p = 0.0025
    z = p * e
    if z > 0: z = min(z, 0.3)
    if z < 0: z = max(z, -0.3)
    return z



def amcl_pose_callback(pose):
    global current_pose
    current_pose = pose
if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("start")

    frame2 = None
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image2)

    depth2 = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth2)

    print("load")
    dnn_face = FaceDetection()
    print("yolo")
    net_pose = HumanPoseEstimation(device_name="GPU")

    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    topic_imu = "/imu/data"
    _imu = None
    rospy.Subscriber(topic_imu, Imu, callback_imu)
    rospy.wait_for_message(topic_imu, Imu)

    current_pose = None
    rospy.Subscriber("/amcl_pose",PoseWithCovarianceStamped,amcl_pose_callback)
    
    olist=[]
    save,check = False,False
    print("start the program")
    while not rospy.is_shutdown():
        rospy.Rate(50).sleep()

        if frame2 is None:
            print("frame2")
            continue
        if depth2 is None:
            print("depth2")
            continue
        if current_pose is None:
            continue
        faces = dnn_face.forward(frame2)
        x,y,a = current_pose.pose.pose.position.x, current_pose.pose.pose.position.y, current_pose.pose.pose.orientation.z
        for (x1,y1,x2,y2) in faces:
            cv2.rectangle(frame2,(x1,y1),(x2,y2),(0,255,0),2)  
            cx,cy = x1+(x2-x1)//2, y1+(y2-y1)//2
            #d = depth2[cy][cx]

            rx,ry,d = get_real_xyz(depth2,cx,cy)
        #print("run") 
            h, w = frame2.shape[:2]
            target_x = x+rx/1000
            target_y = (y+abs((d*d-rx*rx))**0.5)/1000 
            target_z = ry/1000
            '''if (a <0.25 and  a> -0.25) or( (a>=0.75 and a<=1) or  (a<=-0.75 and a>= -1)):
                target_x = x+rx/1000
                target_y = (y+abs((d*d-rx*rx))**0.5)/1000 
            else:
                target_x = (y+abs((d*d-rx*rx))**0.5)/1000 
                target_y = x+rx/1000'''
            if save: 
                olist.append([target_x,target_y,target_z])
                save=False
           # print(rx,ry)
                print(f"s : {[x,y]},{[target_x,target_y,target_z]},{a}")
            if check:
                print(f"c : {[x,y]},{[target_x,target_y,target_z]},{a}")
                check = False
        #print(a)
            
        cv2.imshow("image", frame2)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
        if key in [ord('s'), 27]:
            save = True
        if key in [ord('c'), 27]:
            check = True



