#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.pytorch_models import *
from pcms.openvino_models import Yolov8, HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist
import math
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest
import time
from mr_voice.msg import Voice
from std_msgs.msg import String
from rospkg import RosPack
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import Imu
from typing import Tuple, List
from RobotChassis import RobotChassis
import datetime
from std_srvs.srv import Empty


class FollowMe(object):
    def __init__(self) -> None:
        self.pre_x, self.pre_z = 0.0, 0.0

    def get_pose_target(self, pose, num):
        p = []
        for i in [num]:
            if pose[i][2] > 0:
                p.append(pose[i])

        if len(p) == 0:
            return -1, -1, -1
        return int(p[0][0]), int(p[0][1]), 1

    def get_real_xyz(self, depth, x: int, y: int) -> Tuple[float, float, float]:
        if x < 0 or y < 0:
            return 0, 0, 0
        a1 = 55.0
        b1 = 86.0
        a = a1 * np.pi / 180
        b = b1 * np.pi / 180

        d = depth[y][x]
        h, w = depth.shape[:2]
        if d == 0:
            for k in range(1, 15, 1):
                if d == 0 and y - k >= 0:
                    for j in range(x - k, x + k, 1):
                        if not (0 <= j < w):
                            continue
                        d = depth[y - k][j]
                        if d > 0:
                            break
                if d == 0 and x + k < w:
                    for i in range(y - k, y + k, 1):
                        if not (0 <= i < h):
                            continue
                        d = depth[i][x + k]
                        if d > 0:
                            break
                if d == 0 and y + k < h:
                    for j in range(x + k, x - k, -1):
                        if not (0 <= j < w):
                            continue
                        d = depth[y + k][j]
                        if d > 0:
                            break
                if d == 0 and x - k >= 0:
                    for i in range(y + k, y - k, -1):
                        if not (0 <= i < h):
                            continue
                        d = depth[i][x - k]
                        if d > 0:
                            break
                if d > 0:
                    break
        x = x - w // 2
        y = y - h // 2
        real_y = y * 2 * d * np.tan(a / 2) / h
        real_x = x * 2 * d * np.tan(b / 2) / w
        return real_x, real_y, d

    def calc_linear_x(self, cd: float, td: float) -> float:
        if cd <= 0:
            return 0
        e = cd - td
        p = 0.0005
        x = p * e
        if x > 0:
            x = min(x, 0.15)
        if x < 0:
            x = max(x, -0.15)
        return x

    def calc_angular_z(self, cx: float, tx: float) -> float:
        if cx < 0:
            return 0
        e = tx - cx
        p = 0.0025
        z = p * e
        if z > 0:
            z = min(z, 0.2)
        if z < 0:
            z = max(z, -0.2)
        return z

    def calc_cmd_vel(self, image, depth, cx, cy) -> Tuple[float, float]:
        image = image.copy()
        depth = depth.copy()

        frame = image
        if cx == 2000:
            cur_x, cur_z = 0, 0
            return cur_x, cur_z, frame, "no"

        print(cx, cy)
        _, _, d = self.get_real_xyz(depth, cx, cy)

        cur_x = self.calc_linear_x(d, 800)
        cur_z = self.calc_angular_z(cx, 320)

        dx = cur_x - self.pre_x
        if dx > 0:
            dx = min(dx, 0.15)
        if dx < 0:
            dx = max(dx, -0.15)

        dz = cur_z - self.pre_z
        if dz > 0:
            dz = min(dz, 0.4)
        if dz < 0:
            dz = max(dz, -0.4)

        cur_x = self.pre_x + dx
        cur_z = self.pre_z + dz

        self.pre_x = cur_x
        self.pre_z = cur_z

        return cur_x, cur_z, frame, "yes"


def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)


def get_pose_target2(pose, num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])

    if len(p) == 0:
        return -1, -1, -1
    return int(p[0][0]), int(p[0][1]), 1


def turn_to(angle: float, speed: float):
    global _imu
    max_speed = 1.82
    limit_time = 8
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


def set_gripper(angle, t):
    service_name = "/goal_tool_control"
    rospy.wait_for_service(service_name)

    try:
        service = rospy.ServiceProxy(service_name, SetJointPosition)

        request = SetJointPositionRequest()
        request.joint_position.joint_name = ["gripper"]
        request.joint_position.position = [angle]
        request.path_time = t

        response = service(request)
        return response
    except Exception as e:
        rospy.loginfo("%s" % e)
        return False


def open_gripper(t):
    return set_gripper(0.01, t)


def close_gripper(t):
    return set_gripper(-0.01, t)


def get_pose_target(pose, num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])

    if len(p) == 0:
        return -1, -1
    return int(p[0][0]), int(p[0][1])


def get_real_xyz(dp, x, y, num):
    a1 = 49.5
    b1 = 60.0
    if num == 2:
        a1 = 55.0
        b1 = 86.0
    a = a1 * np.pi / 180
    b = b1 * np.pi / 180
    d = dp[y][x]
    h, w = dp.shape[:2]
    if d == 0:
        for k in range(1, 15, 1):
            if d == 0 and y - k >= 0:
                for j in range(x - k, x + k, 1):
                    if not (0 <= j < w):
                        continue
                    d = dp[y - k][j]
                    if d > 0:
                        break
            if d == 0 and x + k < w:
                for i in range(y - k, y + k, 1):
                    if not (0 <= i < h):
                        continue
                    d = dp[i][x + k]
                    if d > 0:
                        break
            if d == 0 and y + k < h:
                for j in range(x + k, x - k, -1):
                    if not (0 <= j < w):
                        continue
                    d = dp[y + k][j]
                    if d > 0:
                        break
            if d == 0 and x - k >= 0:
                for i in range(y + k, y - k, -1):
                    if not (0 <= i < h):
                        continue
                    d = dp[i][x - k]
                    if d > 0:
                        break
            if d > 0:
                break

    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d


def set_joints(joint1, joint2, joint3, joint4, t):
    service_name = "/goal_joint_space_path"
    rospy.wait_for_service(service_name)

    try:
        service = rospy.ServiceProxy(service_name, SetJointPosition)

        request = SetJointPositionRequest()
        request.joint_position.joint_name = [
            "joint1", "joint2", "joint3", "joint4"]
        request.joint_position.position = [joint1, joint2, joint3, joint4]
        request.path_time = t

        response = service(request)
        return response
    except Exception as e:
        rospy.loginfo("%s" % e)
        return False


def move_to(x, y, z, t):
    service_name = "/goal_task_space_path_position_only"
    rospy.wait_for_service(service_name)

    try:
        service = rospy.ServiceProxy(service_name, SetKinematicsPose)

        request = SetKinematicsPoseRequest()
        request.end_effector_name = "gripper"
        request.kinematics_pose.pose.position.x = x
        request.kinematics_pose.pose.position.y = y
        request.kinematics_pose.pose.position.z = z
        request.path_time = t

        response = service(request)
        return response
    except Exception as e:
        rospy.loginfo("%s" % e)
        return False


def say(a):
    global publisher_speaker
    publisher_speaker.publish(a)


# callback
def callback_imu(msg):
    global _imu
    _imu = msg


def callback_voice(msg):
    global s
    s = msg.text


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
    _topic_image1 = "/cam2/color/image_raw"
    _sub_up_cam_image = rospy.Subscriber(_topic_image1, Image, callback_image1)

    print("astra depth")
    _depth1 = None
    _topic_depth1 = "/cam2/depth/image_raw"
    _sub_up_cam_depth = rospy.Subscriber(_topic_depth1, Image, callback_depth1)
    #
    print("gemini2 rgb")
    _frame2 = None
    _sub_down_cam_image = rospy.Subscriber("/cam1/color/image_raw", Image, callback_image2)

    print("gemini2 depth")
    _depth2 = None
    _sub_down_cam_depth = rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth2)

    s = ""
    print("cmd_vel")
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    print("speaker")
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)

    print("arm")
    t = 3.0
    open_gripper(t)

    # change model
    print("yolov8")
    Kinda = np.loadtxt(RosPack().get_path("mr_dnn") + "/Kinda.csv")
    dnn_yolo1 = Yolov8("bag_japan", device_name="GPU")
    dnn_yolo1.classes = ['obj']

    # two yolo

    print("pose")
    net_pose = HumanPoseEstimation(device_name="GPU")

    print("waiting imu")
    topic_imu = "/imu/data"
    _imu = None
    rospy.Subscriber(topic_imu, Imu, callback_imu)
    rospy.wait_for_message(topic_imu, Imu)

    print("chassis")
    chassis = RobotChassis()

    _fw = FollowMe()

    print("finish loading, start")
    h, w, c = _image1.shape
    img = np.zeros((h, w * 2, c), dtype=np.uint8)
    img[:h, :w, :c] = _image1
    img[:h, w:, :c] = _frame2
    slocnt = 0
    # u_var
    d, one, mask, key, is_turning = 1, 0, 0, 0, False
    ax, ay, az, bx, by, bz = 0, 0, 0, 0, 0, 0
    pre_x, pre_z, haiya, bruh, lcnt, rcnt, run, p_cnt, focnt = 0.0, 0.0, 0, 0, 0, 0, 0, 0, 1
    pos, cnt_list = [2.77, 1.82, 0.148], []
    pre_s = ""
    # main var
    t, ee, s = 3.0, "", ""
    step = "follow"

    clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)

    action = "follow"
    move_turn = "none"
    # wait for prepare
    print("start")
    time.sleep(10)
    need_position = []
    lr = "middle"
    # var in camera
    px, py, pz, pree_cx, pree_cy, = 0, 0, 0, 0, 0
    posecnt = 0
    move_turn = "turn"
    # senser var
    class_need = 0
    closest_person = None
    joint1, joint2, joint3, joint4 = 0, 0, 0, 0
    set_joints(joint1, joint2, joint3, joint4, 3)
    time.sleep(3)

    queue_people_cnt = 0
    while not rospy.is_shutdown():
        # voice check
        # break
        if s != "" and s != pre_s:
            print(s)
            pre_s = s

        rospy.Rate(10).sleep()
        if _frame2 is None: print("down rgb none")
        if _depth2 is None: print("down depth none")
        if _depth1 is None: print("up depth none")
        if _image1 is None: print("up rgb none")
        if _depth1 is None or _image1 is None or _depth2 is None or _frame2 is None: continue

        cx1, cx2, cy1, cy2 = 0, 0, 0, 0
        detection_list = []

        down_image = _frame2.copy()
        down_depth = _depth2.copy()
        up_image = _image1.copy()
        up_depth = _depth1.copy()
        if step == "get_bag":
            detections = dnn_yolo1.forward(down_image)[0]["det"]
            for i, detection in enumerate(detections):
                x1, y1, x2, y2, score, class_id = map(int, detection)
                score = detection[4]
                cx = (x2 - x1) // 2 + x1
                cy = (y2 - y1) // 2 + y1
                if score > 0.5 and class_id == class_need:
                    detection_list.append([x1, y1, x2, y2, cx, cy])
                    cv2.rectangle(down_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(down_image, (cx, cy), 5, (0, 255, 0), -1)
                    print("bag score:", score)
            A = []
            B = []
            yu = 0
            poses = net_pose.forward(up_image)
            if len(poses) > 0:
                YN = -1
                a_num, b_num = 9, 7
                for issack in range(len(poses)):
                    yu = 0
                    if poses[issack][9][2] > 0 and poses[issack][7][2] > 0:
                        YN = 0

                        a_num, b_num = 9, 7
                        A = list(map(int, poses[issack][a_num][:2]))
                        if (640 >= A[0] >= 0 and 320 >= A[1] >= 0):
                            ax, ay, az = get_real_xyz(up_depth, A[0], A[1], 2)
                            if az <= 2500 and az != 0:
                                yu += 1
                        B = list(map(int, poses[issack][b_num][:2]))
                        if (640 >= B[0] >= 0 and 320 >= B[1] >= 0):
                            bx, by, bz = get_real_xyz(up_depth, B[0], B[1], 2)
                            if bz <= 2500 and bz != 0:
                                yu += 1
                    if yu >= 2:
                        break
            print(A, B)
            if len(A) != 0 and yu >= 2 and len(B) != 0:
                cv2.circle(up_image, (A[0], A[1]), 3, (0, 255, 0), -1)
                cv2.circle(up_image, (B[0], B[1]), 3, (0, 255, 0), -1)
        if step == "get_bag":
            if len(A) != 0 and len(B) != 0:

                bx = A[0]
                if len(detection_list) < 1:
                    print("no bag")

                if len(detection_list) <= 2 and len(detection_list) > 0:
                    sort_detection = sorted(detection_list, key=(lambda x: x[0]))
                    print(detection_list)
                    if len(detection_list) == 1:
                        need_position = sort_detection[0]
                    else:

                        if posecnt == 0:
                            if bx < 320:
                                lr = "left"
                            else:
                                lr = "right"
                            posecnt += 1
                        if lr == "left":
                            need_position = sort_detection[0]
                        else:
                            need_position = sort_detection[1]

                    print(need_position)
                    print("ho")
                    x1, y1, x2, y2, cx2, cy2 = map(int, need_position)
                    cv2.rectangle(down_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(down_image, (cx2, cy2), 5, (0, 0, 255), -1)
                    now = datetime.datetime.now()
                    filename = now.strftime("%Y-%m-%d_%H-%M-%S.jpg")
                    output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                    cv2.imwrite(output_dir + filename, down_image)
                    if move_turn == "turn":

                        h, w, c = down_image.shape
                        x1, y1, x2, y2, cx2, cy2 = map(int, need_position)
                        e = w // 2 - cx2
                        v = 0.001 * e
                        if v > 0:
                            v = min(v, 0.3)
                        if v < 0:
                            v = max(v, -0.3)
                        move(0, v)
                        print(e)
                        if abs(e) <= 3:
                            say("walk")
                            action = "front"
                            move_turn = "none"
                            step = "none"
                            print("ys")
        if action == "front":
            print("front")
            cx, cy = w // 2, h // 2
            for i in range(cy + 1, h):
                if _depth2[cy][cx] == 0 or 0 < _depth2[i][cx] < _depth2[cy][cx]:
                    cy = i
            _, _, d = get_real_xyz(_depth2, cx, cy, 2)
            while d > 0 or abs(e) >= 10:
                _, _, d1 = get_real_xyz(_depth2, cx, cy, 2)
                e = d1 - 400  # number is he last distance
                if e <= 10:
                    break
                v = 0.001 * e
                if v > 0:
                    v = min(v, 0.2)
                if v < 0:
                    v = max(v, -0.2)
                print(d1, e, v)
                move(v, 0)
            print("got there")
            joint1, joint2, joint3, joint4 = 0.087, 0.45, 0.758, -1.4
            set_joints(joint1, joint2, joint3, joint4, 1)
            time.sleep(3)
            action = "grap"
            move_turn = "none"
            step = "none"
        if action == "grap":
            for i in range(10000): move(0.2, 0)
            say("I get it")
            time.sleep(t)

            close_gripper(t)
            time.sleep(2)
            joint1, joint2, joint3, joint4 = 0.087, 0.45, 0.758, -1.4
            set_joints(joint1, joint2, joint3, joint4, 1)
            time.sleep(t)

            joint1, joint2, joint3, joint4 = 0.087, 0.119, 0.365, -1.4
            set_joints(joint1, joint2, joint3, joint4, t)
            time.sleep(t)
            joint1, joint2, joint3, joint4 = 0.087, 0.0, 0.365, -1.4
            set_joints(joint1, joint2, joint3, joint4, t)
            time.sleep(t)
            time.sleep(3)
            say("I will follow you now")
            for i in range(50000): move(-0.2, 0)
            action = "follow"

        if action == "follow":
            print('follow')
            msg = Twist()
            poses = net_pose.forward(up_image)
            min_d = 9999
            for i, pose in enumerate(poses):
                if pose[5][2] == 0 or pose[6][2] == 0:
                    continue
                p5 = list(map(int, pose[5][:2]))
                p6 = list(map(int, pose[6][:2]))
                if d >= 1800 or d == 0: continue
                cx = (p5[0] + p6[0]) // 2
                cy = (p5[1] + p6[1]) // 2
                cv2.circle(up_image, p5, 5, (0, 0, 255), -1)
                cv2.circle(up_image, p6, 5, (0, 0, 255), -1)
                cv2.circle(up_image, (cx, cy), 5, (0, 255, 0), -1)
                _, _, d = get_real_xyz(up_depth, cx, cy, 2)

                if (d != 0 and d < min_d) and d <= 1800:
                    t_idx = i
                    min_d = d
            x, z = 0, 0
            if min_d != 9999:
                p5 = list(map(int, poses[t_idx][5][:2]))
                p6 = list(map(int, poses[t_idx][6][:2]))
                cx = (p5[0] + p6[0]) // 2
                cy = (p5[1] + p6[1]) // 2
                _, _, d = get_real_xyz(up_depth, cx, cy, 2)
                cv2.circle(up_image, (cx, cy), 5, (0, 255, 255), -1)
                print("people_d", d)
                if d >= 1800 or d == 0: continue

                x, z, up_image, yn = _fw.calc_cmd_vel(up_image, up_depth, cx, cy)
                print("turn_x_z:", x, z)
            move(x, z)
        if step == "check_voice":
            s = s.lower()
            if "thank" in s or "you" in s:
                action = "none"
                say("I will go back now, bye bye")
                joint1, joint2, joint3, joint4 = 0.000, 0.0, 0, 1.5
                set_joints(joint1, joint2, joint3, joint4, 1)
                time.sleep(t)
                open_gripper(t)
                time.sleep(3)
                joint1, joint2, joint3, joint4 = 0.000, -1.0, 0.3, 0.70
                set_joints(joint1, joint2, joint3, joint4, 1)

                time.sleep(2.5)
                joint1, joint2, joint3, joint4 = 1.7, -1.052, 0.376, 0.696
                set_joints(joint1, joint2, joint3, joint4, 3)

                time.sleep(3)
                action = "back3"
                step = "none"
        if action == "back3":
            break
            clear_costmaps
            # chassis.move_to(-0.703,-4.65,0)
            chassis.move_to(5.51, 6.74, 0)
            # checking
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
            time.sleep(1)
            clear_costmaps
            # chassis.move_to(-2.36,3.43,0)
            chassis.move_to(9.21, 5.51, 0)
            # checking
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
            time.sleep(1)
            clear_costmaps
            # chassis.move_to(-0.531,-2.99,0.201)
            chassis.move_to(7.95, 8.08, 0)
            # checking
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
            time.sleep(1)
            break
        h, w, c = up_image.shape
        upout = cv2.line(up_image, (320, 0), (320, 500), (0, 255, 0), 5)
        downout = cv2.line(down_image, (320, 0), (320, 500), (0, 255, 0), 5)
        img = np.zeros((h, w * 2, c), dtype=np.uint8)
        img[:h, :w, :c] = upout
        img[:h, w:, :c] = downout
        cv2.imshow("frame", img)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
