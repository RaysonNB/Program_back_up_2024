#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge
from pcms.openvino_models import Yolov8, FaceDetection, HumanPoseEstimation, PersonAttributesRecognition
from geometry_msgs.msg import Twist
from RobotChassis import RobotChassis  # 導航
from std_msgs.msg import String
from mr_voice.msg import Voice
import time
import os
import math
from tf.transformations import euler_from_quaternion
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests


def callBack_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def say(text):
    os.system(f'espeak "{text}"')
    rospy.loginfo(text)


def callback_voice(msg):
    global _voice
    _voice = msg


def callBack_depth(msg):
    global _depth
    _depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")

def imu_callback(msg):
    global imu
    imu = msg
    
def PID_z(cx):
    ez = 320 - cx
    p = 0.0025
    z = ez * p
    return max(min(z, 0.6), -0.6)


def PID_x(depth):
    ed = depth - 640
    p = 0.00075
    x = ed * p
    return max(min(x, 0.28), -0.28)


def theNearest(clist):
    n = 10000
    target = 0
    if len(clist) == 0: return -1, -1
    for i in range(len(clist)):
        cx, cy = clist[i][0], clist[i][1]
        # print(cx,cy)
        d = _depth[int(cy)][int(cx)]
        if d < n and d != 0:
            n = d
            target = i
    if target > len(clist) - 1: return -1, -1
    return clist[target], target


def findBody(image):
    pList = dnn_pose.forward(image)
    target,e = -1,1000
    if len(pList) == 0: return []
    for pose in pList:
        cx,cy = pose[5][0],pose[5][1]
        if cx < 320: 
            if 320-cx <e: target,e = pose,320-e
        elif cx >= 320:
            if cx-320 < e: target,e = pose,e-320
    return target
    #_, i = theNearest(checkL)

def findNearestFace(image):
    flist = dnn_face.forward(image)
    _, i = theNearest(flist)
    if i == -1: return -1, -1, -1, -1
    return flist[i]


def count_color(frame):
    h, w, c = frame.shape
    c = 0
    for x in range(w):
        for y in range(h):
            if frame[y, x, 0] != 0 and frame[y, x, 1] != 0 and frame[y, x, 1] != 0:
                c += 1
    return c


def detect_color(frame):
    _frame = cv2.resize(frame, (40, 30))
    hsv_frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2HSV)
    clist = []
    global color
    for c in color:
        mask = cv2.inRange(hsv_frame, np.array(c[1]), np.array(c[2]))
        result = cv2.bitwise_and(_frame, _frame, mask=mask)
        clist.append([count_color(result), c[0]])
    # print(f"colorList:{sorted(clist, reverse=True)}")
    return sorted(clist, reverse=True)[0][1]


def countColorList(L):
    global color
    c, result = -1, ""
    for item in color:
        if c < L.count(item[0]):
            result = item[0]
            c=L.count(item[0])
    return result


def get_real_xyz(x, y, d):
    global _depth
    if _depth is None:
        return -1, -1, -1
    h, w = _depth.shape[:2]
    # d = _depth[y][x]
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    real_y = (h / 2 - y) * 2 * d * np.tan(a / 2) / h
    real_x = (w / 2 - x) * 2 * d * np.tan(b / 2) / w
    return real_x, real_y


def color_filter(frame, hsv_frame, low_range, high_range) -> int:
    low = np.array(low_range)
    high = np.array(high_range)
    color_mask = cv2.inRange(hsv_frame, low, high)
    cnt = cv2.countNonZero(color_mask)
    return cnt


def colorDetectV2(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colorList = ["red", "orange", "yellow", "green", "blue", "purple", "black", "white", "gray"]
    red = color_filter(image, hsv_image, [120, 254, 200], [179, 255, 255])
    orange = color_filter(image, hsv_image, [5, 75, 0], [21, 255, 255])
    yellow = color_filter(image, hsv_image, [22, 93, 0], [35, 255, 255])
    green = color_filter(image, hsv_image, [25, 52, 72], [102, 255, 255])
    blue = color_filter(image, hsv_image, [94, 80, 2], [126, 255, 255])
    purple = color_filter(image, hsv_image, [133, 43, 46], [155, 255, 255])
    black = color_filter(image, hsv_image, [0, 0, 0], [179, 100, 60])
    white = color_filter(image, hsv_image, [0, 0, 135], [172, 111, 255])
    gray = color_filter(image, hsv_image, [0, 0, 61], [172, 111, 134])
    max_color_index = np.argmax(np.array([red, orange, yellow, green, blue, purple, black, white, gray]))
    return colorList[max_color_index]

def find_face(Facelist, status):
    global _depth
    if status == "min": target,mx=0,640
    if status =="max": target,mx = 0,0
    for i in range(len(Facelist)):
        d = _depth[Facelist[i][1]][Facelist[i][0]]
        if d <2500 and d!=0:
            if Facelist[i][0] < mx and status=="min": target,mx =i,Facelist[i][0]
            elif Facelist[i][0] > mx and status=="max": target,mx =i,Facelist[i][0]
    return target


if __name__ == "__main__":
    rospy.init_node("task2")
    rospy.loginfo("program start")

    _image = None
    rospy.Subscriber("/cam2/color/image_raw", Image, callBack_image)
    # rospy.wait_for_message("/cam2/color/image_raw",Image)
    rospy.loginfo("camera finish")

    _depth = None
    rospy.Subscriber("/cam2/depth/image_raw", Image, callBack_depth)

    speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    _voice = None
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    imu = None
    rospy.Subscriber("/imu/data",Imu,imu_callback)
    path_openvino = "/home/pcms/models/openvino"
    dnn_face = FaceDetection(path_openvino)
    dnn_pose = HumanPoseEstimation(path_openvino)

    dnn_feature = PersonAttributesRecognition(path_openvino)
    dnn_glasses = Yolov8("glasses", device_name="GPU")
    #dnn_glasses = Yolov8("best_V1", device_name="GPU")
    dnn_glasses.classes = ['glasses']
    dnn_mask = Yolov8("maskV2", device_name="GPU")
    #dnn_glasses = Yolov8("best_V1", device_name="GPU")
    dnn_mask.classes = ['mask']
    dnn_gender = Yolov8("gender", device_name="GPU")
    dnn_gender.classes = ['female', 'male']

    pub_cmd = rospy.Publisher("/cmd_vel/", Twist, queue_size=10)
    msg_cmd = Twist()
    # h=pixelH
    pos = {"roomG": (-0.314, -7.291, -3.058), "roomM": (5.128, -6.114, 1.619), "Door": (1.772, -7.069, -3.118),"third":{1.772, -7.069, -1}}  # R0 L3.4 M1.8
    color = [["red", [175, 43, 46], [180, 255, 255]], ["orange", [0, 140, 100], [20, 255, 255]],
             ["yellow", [22, 93, 0], [33, 255, 255]],
             ["green", [34, 20, 0], [94, 255, 255]], ["blue", [94, 40, 2], [126, 255, 255]],
             ["purple", [130, 43, 46], [145, 255, 255]],
             ["pink", [125, 100, 30], [165, 255, 255]], ["white", [0, 0, 10], [25, 30, 255]],
             ["black", [22, 0, 0], [180, 255, 70]],["brown",[20, 20, 0],[33, 140, 170]]]

    status, cfinish, name = 0, 1, ""  # 完成次數 正式
    
    nameList= [["Sophie", "selfie"], ["Fleur", "villa", "fuller", "floor", "flood"], ["Kevin","Ivan"], ["Julia"], ["Gabrielle", "Gabriella"], ["Jesse", "Jessie", "Jessa","Jess"], ["Emma"], ["Robin", "Ruben", "moving", "Reuben"], ["Noah"], ["Sara", "Sarah"], ["John"], ["Harrie", "Harry","Harvey", "hardly"], ["Laura","Lara"], ["Liam", "lion", "Leanne"], ["Peter", "Peeta", "pizza", "pita"], ["Hayley","Haiti","hey","Hal","Haley"],["Lucas"],["Susan"], ["William","volume","villain","radium"]]
    location=''
    check_feature = 0
    dGlass, isGlass,dMask,isMask = 0, False,0,False
    
    dGender, gender = 0, ""
    upcolor, downcolor = "", ""
    speakList, genderList, glassesList, maskList,UpColorList, DownColorList,YawList = [], [], [], [], [],[],[]
    height, angular,start,start2 = 0, 0,time.time(),time.time()
    q = [imu.orientation.x,imu.orientation.y,imu.orientation.z,imu.orientation.w]
    _, _, origialYaw = euler_from_quaternion(q)

    qList = ["Is the person wear glasses?","Is the person wear mask?","Is this person a male?"]
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    print("start")
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        if _image is None: continue
        # print("yes")
        frame = _image.copy()

        msg_cmd.angular.z = 0
        msg_cmd.linear.x = 0
        if status == 0:  # 移動去目標場地
            if _voice == None: continue
            if "dining room" in _voice.text or "look" in _voice.text:
                if cfinish == 1:
                    pos["roomG"] = (-0.314, -7.291, 1.9)  # L
                elif cfinish == 2:
                    pos["roomG"] = (-0.314, -7.291, 0.7)  # R -1
                if cfinish!=3: RobotChassis().move_to(*pos["Door"])
                if cfinish == 3:RobotChassis().move_to(*pos["third"])
                if cfinish!=3: RobotChassis().move_to(*pos["roomG"])
                _, _, origialYaw = euler_from_quaternion(q)
                say("I arrive")
                status += 1
        elif status == -1:
            if cfinish == 1:
                pos["roomG"] = (1.6, 2.45, 3.6)  # L
            elif cfinish == 2:
                pos["roomG"] = (1.6, 2.45, 0)  # R
            elif cfinish == 3:
                pos["roomG"] = (1.6, 2.45, 1.6)  # M
            pos["Door"] = (1.29, 0.764, 1.56)
            RobotChassis().move_to(*pos["third"])
            #RobotChassis().move_to(*pos["roomG"])
            say("I arrive")
            break

        elif status == 1:  # 檢測人(找)
            faces = dnn_face.forward(frame)
            if len(faces) == 0:
                if cfinish == 1:  # 向左轉
                    msg_cmd.angular.z = -0.1
                elif cfinish == 2:  # 向右轉
                    msg_cmd.angular.z = 0.1
                elif cfinish == 3:
                    msg_cmd.angular.z = 0.1  # 不轉
            else:
                if cfinish == 1: target = find_face(faces,"min")
                elif cfinish ==2: target = find_face(faces,"max")
                elif cfinish ==2: target = find_face(faces,"max")
                x1, y1, x2, y2 = faces[target]
                cx = x1 + (x2 - x1) // 2
                if (cx >= 315 and cx <= 325):  # 若已移到正中間

                    msg_cmd.angular.z = 0
                    status += 1
                    start = time.time()
                    q = [imu.orientation.x,imu.orientation.y,imu.orientation.z,imu.orientation.w]
                    _, _, Yaw = euler_from_quaternion(q)
                    if (Yaw < 2.281 and Yaw> 1.8) or (Yaw>0.891 and Yaw<1.09): location = "small sofa"
                    elif Yaw <1.8 and Yaw>=1.09 : location = "big sofa"
                    else: location = "dining table"  
                    print(f"angle  O{origialYaw} Y{Yaw}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #key_code = cv2.waitKey(1)
                    cv2.imwrite(f"/home/pcms/Desktop/task2/detectperson{cfinish}.jpg", frame)
                else:
                    if abs(PID_z(cx)) < 1.7:
                        msg_cmd.angular.z, angular = PID_z(cx), angular + math.ceil(PID_z(cx) * 12.5 * -1)
                    else:
                        print("too big")
            pub_cmd.publish(msg_cmd)

        elif status == 2:  # 獲取客人特徵，向客人靠
            pose, end = findBody(_image), time.time()
            #print(pose)
            if len(pose) > 0:
		            #if check_feature > -1 and (end - start) < 23:
                if check_feature <5 and (end - start) < 12:
                    fx1, fy1, fx2, fy2 = list(map(int, findNearestFace(_image)))
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                    if fx1 == -1: continue
                    # print(fx1,fy1,fx2,fy2)
                    faceI = _image[int(fy1 - fy1 * 0.4):int(fy2 + fy2 * 0.1),
                            int(fx1 - fx1 * 0.05):int(fx2 + fx2 * 0.05)]
                    w, h = int(fy2 + fy2 * 0.1) - int(fy1 - fy1 * 0.4), int(fx2 + fx2 * 0.05) - int(fx1 - fx1 * 0.05)
                    #if h < 237:
                        #faceI = cv2.resize(faceI, (w * (237 // h + 1), h * (237 // h + 1)))
                    for i in range(len(qList)):
                        encoding = processor(faceI, qList[i], return_tensors="pt")

                        # forward pass
                        outputs = model(**encoding)
                        logits = outputs.logits
                        idx = logits.argmax(-1).item()
                        #print(f"{text}:", model.config.id2label[idx])
                        if i== 0: glassesList.append(model.config.id2label[idx])
                        if i== 1: maskList.append(model.config.id2label[idx])
                        if i == 2: genderList.append(model.config.id2label[idx])

                    '''
                    # glass
                    faceI_rgb = cv2.cvtColor(faceI, cv2.COLOR_BGR2RGB)
                    dGlass = dnn_glasses.forward(faceI)[0]["det"]
                    if len(dGlass) > 0:
                        gx1, gy1, gx2, gy2, score, class_id = map(int, dGlass[0])
                        print(f"score={dGlass[0][4]}")
                        if dGlass[0][4] > 0.2: isGlass = True

                    # gender
                    dGender = dnn_gender.forward(faceI)[0]["det"]
                    if len(dGender) > 0:
                        # print(f"dgend:{dGender}")
                        gx1, gy1, gx2, gy2, score, class_id = map(int, dGender[0])
                        if class_id == 0: genderList.append("female")
                        if class_id == 1: genderList.append("male")
                        # print(f"id:{class_id}, sc:{dGender[0][4]}")

                    #mask
                    #faceI_rgb = cv2.cvtColor(faceI, cv2.COLOR_BGR2RGB)
                    dMask = dnn_mask.forward(faceI)[0]["det"]
                    if len(dMask) > 0:
                        gx1, gy1, gx2, gy2, score, class_id = map(int, dMask[0])
                        print(f"score={dMask[0][4]}")
                        if dMask[0][4] > 0.2: isMask = True'''
                    check_feature += 1

                    # color
                    if len(pose) >= 14:
                        ux1, uy1, ux2, uy2, dx1, dy1, dx2, dy2 = int(pose[6][0]), int(pose[6][1]), int(
                            pose[11][0]), int(pose[11][1]), int(pose[12][0]), int(pose[12][1]), int(pose[13][0]), int(
                            pose[13][1])
                        # print(ux1,uy1,ux2,uy2,dx1,dy1,dx2,dy2)
                        if ux1 != 0 and uy1 != 0 and ux2 != 0 and uy2 != 0 and ux1 < ux2 and uy1 < uy2:
                            # upcolor = colorDetectV2(frame[uy1:uy2,ux1:ux2])
                            upcolor = detect_color(frame[uy1:uy2, ux1:ux2])
                            UpColorList.append(upcolor)
                            cv2.imshow("up", frame[uy1:uy2, ux1:ux2])
                            # cv2.imshow("up",body[uy1:int(uy2-uy2*0.2),ux1:ux2])
                            print(f"up:{upcolor}")
                        if dx1 != 0 and dy1 != 0 and dx2 != 0 and dy2 != 0 and dx1 < dx2 and dy1 < dy2:
                            # downcolor = colorDetectV2(frame[dy1:dy2,int(dx1-dx1*0.8):int(dx2+dx2*0.8)])
                            downcolor = detect_color(frame[dy1:dy2, int(dx1 - dx1 * 0.1):int(dx2 + dx2 * 0.1)])
                            DownColorList.append(downcolor)
                            cv2.imshow("down", frame[dy1:dy2, int(dx1 - dx1 * 0.1):int(dx2 + dx2 * 0.1)])
                            print(f"down:{downcolor}")
                        key_code = cv2.waitKey(1)

                    # height
                    '''fx1, fy1, fx2, fy2 = list(map(int, findFace(_image)))
                    cx, cy = fx1 + (fx2 - fx1) // 2, fy1 + (fy2 - fy1) // 2
                    if fx1 != -1:
                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                        key_code = cv2.waitKey(1)
                        cv2.imwrite(f"/home/pcms/Desktop/task2/detectperson{cfinish}.jpg", frame)
                    d = _depth[cy][cx]
                    rx, ry = get_real_ xyz(fx1 + (fx
                    2 - fx1) // 2, fy1, d)
                    # print(nx,ny)
                    print(f"realY:{ry},height:{106 + ry / 10},d:{d}")
                    heightList.append(106 + ry / 10)
                    
                    start2 = time.time()'''
                    start2 = time.time()
                    
                else:
                    print(f"isGlass:{isGlass}")
                    print(f"isMask:{isMask}") 
                    if genderList.count("yes") > genderList.count("no"): gender = "male"
                    else: gender = "female"
                    if glassesList.count("yes") > glassesList.count("no"): isGlass = True
                    else: isGlass = False
                    if maskList.count("yes") > maskList.count("no"): isMask = True
                    else: isMask = False
                    
                    #height = int(sum(heightList) / 5)
                    print(f"gender:{gender}")
                    upcolor, downcolor = countColorList(UpColorList), countColorList(DownColorList)
                    print(f"upcolor:{upcolor},downcolorL{downcolor}")
                    
                    ux1, uy1, ux2, uy2 = int(pose[6][0]), int(pose[6][1]), int(pose[11][0]), int(pose[11][1])
                    end2 = time.time()
                    if (end2 - start2) > 15 and check_feature>1:
                        status+=1

                    if ux1 != 0 and uy1 != 0 and ux2 != 0 and uy2 != 0 and ux1 < ux2 and uy1 < uy2:
                        cv2.rectangle(frame, (ux1, uy1), (ux2, uy2), (0, 255, 0), 2)
                        d = _depth[uy1 + (uy2 - uy1) // 2][ux1 + (ux2 - ux1) // 2]
                        print(d)
                        if d != 0:
                            if (d <= 870 or d > 765):
                                v = PID_x(d)
                                if abs(v) < 1.5:
                                    msg_cmd.linear.x = v
                                else:
                                    print("too big")
                            else:
                                msg_cmd.linear.x = 0
                                status += 1
                                
                            pub_cmd.publish(msg_cmd)

        elif status == 3:  # 問嘢
            
            say("What is your name?")
            start = time.time()
            status += 1
            _voice,name = None,""
            

        elif status == 4:  # 收名
            end = time.time()
            if (end - start) > 15:
                status += 1
                name = "Unknown"
            if _voice == None: continue
            ans = _voice.text.split()
            print(ans)
            if "name" in _voice.text:
                for i in nameList:
                    for n in i:
                        if n == ans[-1]:
                            name=i[0]
            else:
                if len(ans) == 1:
                    name = ans
            print(f"{name=}")
            status += 1

        elif status == 5:  # 回去
            RobotChassis().move_to(*pos["Door"])
            RobotChassis().move_to(*pos["roomM"])
            status += 1

        elif status == 6:  # 報告
            c = 0
            if cfinish == 1: s = "first"
            if cfinish == 2: s = "second"
            if cfinish == 3: s = "third"
            say(f"I found the {s} guest")
            say(f"The name of the guest is {name}")
            say(f"{name} is near to {location}")
            
            if gender != "" and c < 2 and not "hairtype" in speakList and gender== "female":
                say(f"{name} has a long hair")
                c += 1
                speakList.append("hairtype")
            if c < 2 and not "gender" in speakList and gender != "":
                say(f"{name} is a {gender}")
                speakList.append("gender")
                c += 1
                
            if isGlass == True and c < 2 and not "glasses" in speakList:
                say(f"{name} wear a glasses")
                speakList.append("glasses")
                c += 1
            if isMask == True and c < 2 and not "mask" in speakList:
                say(f"{name} wear a mask")
                speakList.append("mask")
                c += 1
            if upcolor != "" and c < 2 and not "upcolor" in speakList:
                say(f"{name} wear a {upcolor} cloth")
                speakList.append("upcolor")
                c += 1
            if downcolor != "" and c < 2 and not "downcolor" in speakList:
                say(f"{name} wear a {downcolor} pant")
                speakList.append("downcolor")
                c += 1

            isGlass,isMask, gender, height, upcolor, downcolor, _voice, name = False,False, "", 0, "", "", None,""
            heightList, genderList, glassesList,maskList,UpColorList, DownColorList = [], [], [], [], [], []
            cfinish, status, check_feature = cfinish + 1, 0, 0
            if cfinish > 3: 
                say("thank you")
                break
        cv2.imshow("image", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
rospy.loginfo("task2 end")