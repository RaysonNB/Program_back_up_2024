#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import Yolov8, HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist
from pcms.pytorch_models import *
from pcms.openvino_yolov8 import *
import math
import time
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
import subprocess
import datetime
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


def test_point(xs, ys, d):
    sty=ys
    stx=xs
    if sty * math.sin(d) < 0:
        n1x = -abs(sty * math.sin(d)) + stx * math.cos(d)
    else:
        n1x = abs(sty * math.sin(d)) + stx * math.cos(d)
    
    n1y = ((stx ** 2 + sty ** 2) * math.cos(d) - stx * n1x) / sty
    return n1x,n1y


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("Brian Nigga")

    frame2 = None
    rospy.Subscriber("/cam1/color/image_raw", Image, callback_image2)

    depth2 = None
    rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth2)

    print("load")
    dnn_yolo = Yolov8("yolov8n", device_name="GPU")
    print("yolo")
    net_pose = HumanPoseEstimation(device_name="GPU")

    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    topic_imu = "/imu/data"
    _imu = None
    rospy.Subscriber(topic_imu, Imu, callback_imu)
    rospy.wait_for_message(topic_imu, Imu)

    step = "get"  # remember
    f_cnt = 0
    step2 = "dead"  # remember
    ax, ay, az, bx, by, bz = 0, 0, 0, 0, 0, 0
    b1, b2, b3, b4, b5 = 0, 0, 0, 0, 0
    pre_z, pre_x = 0, 0
    cur_z, cur_x = 0, 0
    axs1, ays1, axs2, ays2=0,0,0,0
    test = 0
    p_list = []
    sb = 0
    framecnt = 0
    bottlecolor = ["blue", "orange", "pink"]
    saidd = 0
    get_b = 0
    bottlecnt = 0
    line_destory_cnt = 0
    degree666=90
    #turn(0)
    getframeyyy="person"
    times_cnt = 1  # times count
    # cv2.imshow("bottle", frame2)
    pose_cnt_cnt=0
    say("start the program")
    while not rospy.is_shutdown():
        rospy.Rate(50).sleep()

        if frame2 is None:
            print("frame2")
            continue
        if depth2 is None:
            print("depth2")
            continue
        line_frame = frame2.copy()
        line_img = np.zeros_like(line_frame)
        frame2 = frame2.copy()
        bottle = []
        capture_img = frame2.copy()
        
        t_pose,flag = None,None
        E,ggg,TTT,ind,sumd = 0,0,0,0,0
        s_d,s_c,dis_list,al,points = [],[],[],[],[]
        
        outframe = frame2.copy()
        az,bz=0,0
        hand_cnt=0
        if getframeyyy=="person":
            A=[]
            B=[]
            ax,ay,az,bx,by,bz=0,0,0,0,0,0
            ys_cnt=0
            poses = net_pose.forward(outframe)
            if len(poses) > 0:
                YN = -1
                a_num, b_num = 10,8
                if poses[0][10][2] > 0 and poses[0][8][2] > 0:
                    YN = 0
                    a_num, b_num = 10,8
                    A = list(map(int, poses[0][a_num][:2]))
                    if(640>=A[0]>=0 and 320>=A[1]>=0):
                        ax,ay,az = get_real_xyz(depth2, A[0], A[1])
                    B = list(map(int, poses[0][b_num][:2]))
                    if(640>=B[0]>=0 and 320>=B[1]>=0):
                        bx,by,bz = get_real_xyz(depth2, B[0], B[1])
            print(A, B)
            if len(A) != 0 and az!=0 and az<2500:
                cv2.circle(outframe, (A[0], A[1]), 3, (0, 0, 255), -1)
                ys_cnt+=1
            if len(B) != 0 and bz!=0 and bz<2500:
                cv2.circle(outframe, (B[0], B[1]), 3, (0, 0, 255), -1)
                ys_cnt+=1
            print("hand",hand_cnt)
            cv2.putText(outframe, str(hand_cnt), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 2)
            g=outframe.copy()
            cv2.imshow("hand", g)
            
            if ax!=0 and bx!=0 and bz!=0 and az!=0 and by!=0 and ay!=0 and ys_cnt>=2:
                hand_cnt+=1
            if hand_cnt>=5:
                #say("hand "+str(angle))
                print("before position",ax, ay,az,bx, by,bz)
                pose_cnt_cnt+=1
                getframeyyy = "object"
                #time.sleep(1)
                print("bext")
                say("found hand")
                turn(-degree666+7)
                print("before position",ax, ay,az,bx, by,bz)
                
        if getframeyyy=="object":
        
            axs1, azs1 = test_point(ax, az, degree666)
            bxs1, bzs1 = test_point(bx, bz, degree666)
            detections = dnn_yolo.forward(outframe)[0]["det"]
            for i, detection in enumerate(detections):
                x1, y1, x2, y2, score, class_id = map(int, detection)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                score = detection[4]
                if class_id != 39: continue
                if score < 0.4: continue
                if cy>=480: continue
                print(cx,cy,"position bottle")
                cx=min(cx,640)
                cy=min(cy,480-1)
                k2, kk1, kkkz = get_real_xyz(depth2, cx, cy)
                if kkkz > 2500 or abs(kkkz) <= 0: continue
                al.append([x1, y1, x2, y2, score, class_id])
                # print(float(score), class_id)
                hhh=str(class_id)+" " +str(k2)+" "+str(kk1)+" "+str(kkkz)
                cv2.rectangle(outframe, (x1, y1), (x2, y2), (0, 255, 0), 5)
                cv2.putText(outframe, str(hhh), (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            bb = sorted(al, key=(lambda x: x[0]))
            for i, detection in enumerate(bb):
                # print(detection)
                x1, y1, x2, y2, score, class_id = map(int, detection)
                score = detection[4]
                # print(id)
                ggg = 1
                bottle.append(detection)
                E += 1
                cx1 = (x2 - x1) // 2 + x1
                cy1 = (y2 - y1) // 2 + y1

                px, py, pz = get_real_xyz(depth2, cx1, cy1)
                dis_list.append(pz)
                cnt = get_distance(px, py, pz, axs1, ay, azs1, bxs1, by, bzs1)
                cv2.putText(outframe, str(int(cnt) // 10), (x1 + 5, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.15,
                                    (0, 0, 255), 2)
                cnt = int(cnt)
                '''
                if cnt != 0 and cnt <= 600:
                    cnt = int(cnt)
                else:
                    cnt = 9999'''
                s_c.append(cnt)
                s_d.append(pz)

            if ggg == 0: s_c = [9999]
            TTT = min(s_c)
            E = s_c.index(TTT)
            for i, detection in enumerate(bottle):
                # print("1")
                x1, y1, x2, y2, score, class_id = map(int, detection)
                
                if (class_id == 39):
                    if i == E:
                        cx1 = (x2 - x1) // 2 + x1
                        cy1 = (y2 - y1) // 2 + y1
                        cv2.putText(outframe, str(int(TTT) // 10), (x1 + 5, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.15,
                                    (0, 0, 255), 2)
                        cv2.rectangle(outframe, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        if i == 0: b1 += 1
                        if i == 1: b2 += 1
                        if i == 2: b3 += 1
                        _, _, dddd1 = get_real_xyz(depth2, cx1, cy1)

                        break

                    else:
                        v = s_c[i]
                        
                        cv2.putText(outframe, str(int(v)), (x1 + 5, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                    2)
            #if b1 == max(b1, b2, b3): mark = 0
            #if b2 == max(b1, b2, b3): mark = 1
            #if b3 == max(b1, b2, b3): mark = 2
            #times_cnt = 1
            #if b1 >= times_cnt or b2 >= times_cnt or b3 >= times_cnt:
            #    b1, b2, b3 = 0, 0, 0
            print("b1: %d b2: %d b3: %d" % (b1, b2, b3))
        frame2 = cv2.resize(outframe, (640 * 2, 480 * 2))
        cv2.imshow("image", frame2)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break



