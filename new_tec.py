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
from gtts import gTTS
from playsound import playsound
from mr_voice.msg import Voice
from std_msgs.msg import String



def say(text):
    voice="en"
    speed=150
    os.system(f"espeak -v{voice} -s{speed} '{text}'")

# gemini2
def callback_image2(msg):
    global frame2
    frame2 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth2(msg):
    global depth2
    depth2 = CvBridge().imgmsg_to_cv2(msg, "passthrough")


# imu chassis
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
    return real_x, real_y, d + 200

def get_real_xyz2(dp, x, y):
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
    max_speed = 0.01
    limit_time = 1
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
        # print(yaw, e)
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
    angle = angle * 180 / math.pi
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
    if x > 0: x = min(x, 0.2)
    if x < 0: x = max(x, -0.2)
    return x


def calc_angular_z(cx, tx):
    if cx < 0: return 0
    e = tx - cx
    p = 0.0025
    z = p * e
    if z > 0: z = min(z, 0.3)
    if z < 0: z = max(z, -0.3)
    return z

'''
def test_point(xs, ys, d):
    d = d * math.pi / 180
    ys1 = math.cos(d) * ys + math.sin(d) * xs  # 12 to 01
    ys2 = math.cos(d) * ys - math.sin(d) * xs  # 01 to 12
    if xs != 0:
        xs1 = (math.cos(d) * (xs ** 2 + ys ** 2) - ys * ys1) / xs
        xs2 = (math.cos(d) * (xs ** 2 + ys ** 2) - ys * ys2) / xs
    else:
        xs1 = -1 * math.sqrt(ys ** 2 - ys1 ** 2)
        xs2 = math.sqrt(ys ** 2 - ys2 ** 2)
    return xs1, ys1, xs2, ys2
'''
def test_point(xs, ys, d):
    d = d * math.pi / 180
    
    #ys += 240
    
    ys1 = math.cos(d) * ys + math.sin(d) * xs  # 12 to 01
    ys2 = math.cos(d) * ys - math.sin(d) * xs  # 01 to 12
    
    if xs != 0:
        xs1 = (math.cos(d) * (xs ** 2 + ys ** 2) - ys * ys1) / xs
        xs2 = (math.cos(d) * (xs ** 2 + ys ** 2) - ys * ys2) / xs
    else:
        xs1 = -1 * math.sqrt(ys ** 2 - ys1 ** 2)
        xs2 = math.sqrt(ys ** 2 - ys2 ** 2)
    return xs1, ys1, xs2, ys2
    

def callback_voice(msg):
    global s
    s = msg.text




if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("dead")

    frame2 = None
    rospy.Subscriber("/cam1/color/image_raw", Image, callback_image2)

    depth2 = None
    rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth2)

    print("load")
    dnn_yolo = Yolov8("yolov8n", device_name="GPU")
    print("yolo")
    net_pose = HumanPoseEstimation(device_name="GPU")
    print("pose")
    s = ""
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)

    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    topic_imu = "/imu/data"
    _imu = None
    rospy.Subscriber(topic_imu, Imu, callback_imu)
    rospy.wait_for_message(topic_imu, Imu)

    # step = "get"  # remember
    f_cnt = 0
    # step2 = "dead"  # remember
    ax, ay, az, bx, by, bz = 0, 0, 0, 0, 0, 0
    b1, b2, b3, b4, b5 = 0, 0, 0, 0, 0
    pre_z, pre_x = 0, 0
    cur_z, cur_x = 0, 0
    axs1, ays1, axs2, ays2 = 0, 0, 0, 0
    test = 0
    p_list = []
    sb = 0
    best_num = 40
    framecnt = 0
    bottlecolor = ["blue", "orange", "pink"]
    saidd = 0
    get_b = 0
    bottlecnt = 0
    line_destory_cnt = 0
    degree666 = 90
    step2 = 'none'
    getframeyyy = "person"
    times_cnt = 1  # times count
    # cv2.imshow("bottle", frame2)
    pose_cnt_cnt = 0
    print("start the program")
    hand_cnt = 0
    game = "turn"
    turn_angle = -30
    yooooo = 0
    mode = 0
    ppp_cnt = 0
    get1 = 0
    say_degree = 0
    t = 3.0
    mark=0
    b1,b2,b3=0,0,0
    check_bootle_cnt = 0
    
    mark=-1
    give_mode=0
    
    say_help=0
    #say("start the program")
    while not rospy.is_shutdown():
        rospy.Rate(50).sleep()

        if frame2 is None:
            print("frame2")
            continue
        if depth2 is None:
            print("depth2")
            continue
        if _imu is None:
            print("_imu")
            continue
        frame2 = frame2.copy()
        bottle = []

        t_pose, flag = None, None
        E, ggg, TTT, ind, sumd = 0, 0, 0, 0, 0
        s_d, s_c, dis_list, al, points = [], [], [], [], []

        outframe = frame2.copy()
        # az, bz = 0, 0
        if getframeyyy == "person":
            az, bz = 0, 0
            A = []
            B = []
            yu = 0
            poses = net_pose.forward(outframe)
            if len(poses) > 0:
                YN = -1
                a_num, b_num = 9,7
                for issack in range(len(poses)):
                    yu=0
                    if poses[issack][9][2] > 0 and poses[issack][7][2] > 0:
                        YN = 0
                        
                        a_num, b_num = 9,7
                        A = list(map(int, poses[issack][a_num][:2]))
                        if (640 >= A[0] >= 0 and 320 >= A[1] >= 0):
                            ax, ay, az = get_real_xyz(depth2, A[0], A[1])
                            if az<=2500 and az!=0:
                                yu += 1
                        B = list(map(int, poses[issack][b_num][:2]))
                        if (640 >= B[0] >= 0 and 320 >= B[1] >= 0):
                            bx, by, bz = get_real_xyz(depth2, B[0], B[1])
                            if bz<=2500 and bz!=0:
                                yu += 1
                    if yu>=2:
                        break
            print(A, B)
            if len(A)!=0 and len(B)!=0 and yu >= 2:
                cv2.circle(outframe, (A[0], A[1]), 3, (255, 255, 0), -1)
                cv2.circle(outframe, (B[0], B[1]), 3, (255, 255, 0), -1)
            if len(A) != 0 and len(B) != 0 and az != 0 and bz != 0:
                hand_cnt += 1
            print("hand",hand_cnt)
            if hand_cnt >= 30:
                # time.sleep(2)
                print("before position", ax, ay, az, bx, by, bz)
                pose_cnt_cnt += 1
                getframeyyy = "_code"
                # turn(degree666)
                say("you looks not feeling well")
                #time.sleep(2)
                say("Do you need any help to feel better?")
                #time.sleep(3)
                say("Rather you need me to clip the bottle on the side?")
                move(0.0, 0.0)
                print("before position", ax, ay, az, bx, by, bz)

        if getframeyyy == "_code":
            if mode == 0 and mode != 1:
                q = [
                    _imu.orientation.x,
                    _imu.orientation.y,
                    _imu.orientation.z,
                    _imu.orientation.w
                ]
                roll1, pitch1, yaw1 = euler_from_quaternion(q)
                mode = 1
            # break
            al = []
            detections = dnn_yolo.forward(outframe)[0]["det"]
            for i, detection in enumerate(detections):
                x1, y1, x2, y2, score, class_id = map(int, detection)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                score = detection[4]
                if class_id != 39: continue
                if score < 0.1: continue
                if cy >= 480: continue
                print(cx, cy, "position bottle")
                cx = min(cx, 640)
                cy = min(cy, 480 - 1)
                if (640 < cx or cx < 0 and 320 < cy or cy < 0): continue

                k2, kk1, kkkz = get_real_xyz(depth2, cx, cy)
                if kkkz > 2500 or abs(kkkz) <= 0: continue
                al.append([x1, y1, x2, y2, score, class_id])
                # print(float(score), class_id)
                hhh = str(class_id) + " " + str(k2) + " " + str(kk1) + " " + str(kkkz)
                cv2.rectangle(outframe, (x1, y1), (x2, y2), (0, 255, 0), 5)
                cv2.putText(outframe, str(hhh), (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            bb = sorted(al, key=(lambda x: x[0]))

            if (len(bb) < 3) and mode == 1:
                move(0, -0.3)

            else:

                mode += 1
                if get1 == 0:
                    mode = 3
                    q = [
                        _imu.orientation.x,
                        _imu.orientation.y,
                        _imu.orientation.z,
                        _imu.orientation.w
                    ]
                    roll2, pitch2, yawb = euler_from_quaternion(q)
                    print("hello")
                    get1 += 1

                yaw2 = yawb
                move(0, 0)
                degree666 = (2 * np.pi - ((yaw2 + np.pi) - (yaw1 + np.pi))) * 180 / np.pi % 360
                #print("turn_degreeeeeeeeeeeee:  ", degree666)
                if say_degree == 0:
                    #say("I turn" + str(int(degree666)) + "degree")
                    say_degree += 1
                print("yaw1-yaw2", yaw1 - yaw2)
                print("yaw1", yaw1)
                print("yaw2", yaw2)
                print("roll1", roll1)
                print("roll2", roll2)
                print("pitch1", pitch1)
                print("pitch2", pitch2)
                axs2, azs2, axs1, azs1 = test_point(ax, az, degree666)
                bxs2, bzs2, bxs1, bzs1 = test_point(bx, bz, degree666)
                print("before position", ax, ay, az, bx, by, bz)
                print("cal hand position", axs2, ay, azs2, bxs2, by, bzs2)
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
                    # pxs2, pzs2, pxs1, pzs1 = test_point(px, pz, degree666)
                    # print("bottle",i+1, pxs2,py, pzs2)
                    cnt = get_distance(px, py, pz, axs2, ay, azs2, bxs2, by, bzs2)
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
                        if i == 0:
                            cx1 = (x2 - x1) // 2 + x1
                            cy1 = (y2 - y1) // 2 + y1

                            cv2.putText(outframe, str(int(TTT) // 10), (x1 + 5, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.15,
                                        (0, 0, 255), 2)
                            cv2.rectangle(outframe, (x1, y1), (x2, y2), (0, 0, 255), 5)
                            _, _, dddd1 = get_real_xyz(depth2, cx1, cy1)

                        else:
                            v = s_c[i]
                            cv2.putText(outframe, str(int(v)), (x1 + 5, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 255, 0),
                                        2)
            if ggg!=0 and step2!="gettttt":
                if E==0: mark=0
                if E==1: mark=1
                if E==2: mark=2
                mark=0
            if len(bb)>=2:
                check_bootle_cnt += 1
            print("check_bootle_cnt", check_bootle_cnt)
            print("getframeyyy", getframeyyy)
            #mark=0
            if mark!=-1 and check_bootle_cnt>=30:
                mark=0
                h, w, c = outframe.shape
                x1, y1, x2, y2, score, class_id = map(int, bb[mark])
                if framecnt == 0:
                    face_box = [x1, y1, x2, y2]
                    box_roi = outframe[face_box[1]:face_box[3] - 1, face_box[0]:face_box[2] - 1, :]
                    fh, fw = abs(x1 - x2), abs(y1 - y2)
                    box_roi = cv2.resize(box_roi, (fh * 10, fw * 10), interpolation=cv2.INTER_AREA)
                    cv2.imshow("bottle", box_roi)
                    get_b = mark
                    framecnt += 1
                #time.sleep(10)
            if sb == 0 and mark!=-1 and check_bootle_cnt>=40:
                #say("you want")
                time.sleep(7)
                mark=0
                #if mark == 0: say("I see you fell down and pointing at the left medicine, it seems that you need to call the ambulence, I will call for help right now.")
                #if mark == 1: say("the middle medicine")
                #if mark == 2: say("the right medicine")
                
                mark=0
                sb += 1
            if len(bb) >= 2 and mark!=-1 and check_bootle_cnt>=45:
                #getframeyyy="person1"
                if say_help==0:
                    say("I see you fell down and pointing at the medicine, it seems that you need to call the ambulence, I will call for help right now.")
                    say_help+=1
                print("say")
                
                #break
                #say("I found the medicine, do you need any further help?")
        E = outframe.copy()

        cv2.imshow("image", E)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break

