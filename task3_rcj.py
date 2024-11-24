#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from RobotChassis import RobotChassis
from pcms.openvino_models import HumanPoseEstimation, Yolov8
from mr_voice.msg import Voice
from std_msgs.msg import String
import yaml, time, cv2, os
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
import numpy as np
from gtts import gTTS
from playsound import playsound
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image as IMAGE
from std_srvs.srv import Empty
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest

class PID:
    def __init__(self, p, i, d) -> None:
        self.p = p
        self.i = i
        self.d = d
        self.last_error = 0
        self.errors = [0]
        self.sum_error = 0
        self.frame = 0
    def control(self, err):
        P = self.p * err
        I = sum(self.errors) / len(self.errors)
        D = self.d*(err - self.last_error)

        self.last_error = err
        self.frame += 1
        self.sum_error -= self.errors[self.frame % 100]
        self.errors[self.frame % 100] = err
        self.sum_error += err

        return P + I + D
"""
pid = PID(0.5, 0.1, 0.3)
while True:
    err = ...
    power = pid.control(err)
    pub.publish(power)
"""
def angular_PID(cx, tx):
    e = tx - cx
    p = 0.0015
    z = p * e
    if z > 0:
        z = min(z, 0.25)
        z = max(z, 0.05)
    if z < 0:
        z = max(z, -0.25)
        z = min(z, -0.05)
    return z

def detect_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv[:, :, 0]], [0], None, [180], [0, 180])
    s = cv2.calcHist([hsv[:, :, 1]], [0], None, [256], [0, 255])
    v = cv2.calcHist([hsv[:, :, 2]], [0], None, [256], [0, 255])
    a = np.reshape(h, 180)
    
    max_color = np.argmax(a) * 2

    s = None
    if 0 <= max_color <= 10 or 156 <= max_color <= 180:
        s = "red"
    elif 11 <= max_color < 25:
        s = "orange"
    elif 26 <= max_color <= 34:
        s = "yellow"
    elif 35 <= max_color <= 77:
        s = "green"
    elif 78 <= max_color <= 124:
        s = "blue"
    elif 125 <= max_color <= 155:
        s = "purple"
    return s

def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth(msg):
    global _depth
    _depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")

def callback_voice(msg):
    global _voice
    _voice = msg

def imu_callback(msg: Imu):
    global imu
    imu = msg

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
    
def arm_control(a,b,c,d):

    joint1, joint2, joint3, joint4 = a,b,c,d
    set_joints(joint1, joint2, joint3, joint4, 3)
    time.sleep(3)
    
 

def say(g):
    os.system(f'espeak "{g}"')
    rospy.loginfo(g)
    '''
    tts = gTTS(g)
    time.sleep(0.5)
    # Save the speech as an audio file
    speech_file = "speech.mp3"
    tts.save(speech_file)
    time.sleep(0.5)
    rospy.loginfo(g)
    # Play the speech
    playsound(speech_file)
    '''
    

def text_to_ans(t, type_of_text):
    name_list = ["Adam", "Axel", "Chris", "Hunter", "Jack", "Max", "Paris", "Robin", "Olivia", "William"]
    possibility_name = [["Adam","Aiden"], ["Axel","ax"], ["Chris", "increase","crease"], ["Hunter","Hunnah"], ["Jack","talk","call"], ["Max","next"], ["Paris", "par", "pan", "pans","pants"], ["Robin","open","Ruben","Rubin"], ["Olivia","sia","Olive","olaf"], ["William","volume","villain","radium"]]

    drink_list = ["coke", "green tea", "wine", "orange juice", "sprite", "soda"]
    possibility_drink = [["coke","talk","call","cook"], ["greentea", "trinity"], ["wine","win","quiet"], ["orangejuice"], ["sprite"], ["soda"]]

    if type_of_text == "name":
        for i,p in enumerate(possibility_name):
            for pp in p:
                if pp.lower() in t.lower():
                    return name_list[i]
    else:
        for i,p in enumerate(possibility_drink):
            for pp in p:
                if pp.lower() in t.lower():
                    return drink_list[i]


def move(forward_speed: float = 0, turn_speed: float = 0):
        global pub_cmd
        msg = Twist()
        msg.linear.x = forward_speed
        msg.angular.z = turn_speed
        pub_cmd.publish(msg)


def turn_to(angle: float, speed: float):
    global imu
    max_speed = 1.82
    limit_time = 8
    start_time = rospy.get_time()
    while True:
        q = [
            imu.orientation.x,
            imu.orientation.y,
            imu.orientation.z,
            imu.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(q)
        e = angle - yaw
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


def body_slice(image, p1, p2):
    global dnn_pose, _depth
    
    poses = dnn_pose.forward(image)
    """points_name = {
        0: "NOSE",
        1: "EYE_L",         2: "EYE_R",
        3: "EAR_L",         4: "EAR_R",
        5: "SHOULDER_L",    6: "SHOULDER_R",
        7: "ELBOW_L",       8: "ELBOW_R",
        9: "WRIST_L",       10:"WRIST_R",
        11:"HIP_L",         12:"HIP_R",
        13:"KNEE_L",        14:"KNEE_R",
        15:"ANKLE_L",       16:"ANKLE_R"
    }
    """
    depth = _depth.copy()
    d = 999999999
    idx = None
    for ix, pose in enumerate(poses):
        min_x, max_x, min_y, max_y = 11111, -1111, 11111, -1111
        for i, p in enumerate(pose):
            x, y, c = map(int, p)
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y
        cx, cy = (min_x+max_x)//2, (min_y+max_y)//2
        if depth[cy][cx] < d:
            d = depth[y][x]
            idx = ix
    
    x1, y1, x2, y2 = int(poses[idx][p1][0]), int(poses[idx][p1][1]), int(poses[idx][p2][0]), int(poses[idx][p2][1])
    x1, x2 = min(x1,x2), max(x1,x2)
    y1, y2 = min(y1, y2), max(y1,y2)
    print(f"{x1=} {x2=} {y1=} {y2=}")

    print(f"{image.shape}")
    return image[y1:y2, x1:x2, :].copy()
    
def color_filter(frame, hsv_frame, low_range, high_range) -> int:
    """
    :param low_range:
    :param high_range:
    :return: the number of pixel that refer to a specific color
    """
    low = np.array(low_range)
    high = np.array(high_range)
    color_mask = cv2.inRange(hsv_frame, low, high)
    # color = cv2.bitwise_and(frame, frame, mask=color_mask)

    cnt = cv2.countNonZero(color_mask)
    return cnt

def color_detect(image) -> str:
    """
    :param image:
    :return: the color that appear in the most arena of the image
    """     
        
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_list = ["red", "orange", "yellow", "green", "blue", "purple", "black", "white", "gray"]

    # find the pixels that refer to the specific color
    red = color_filter(image, hsv_image, [120, 254, 200], [179, 255, 255])

    orange = color_filter(image, hsv_image, [5, 75, 0], [21, 255, 255])

    yellow = color_filter(image, hsv_image, [22, 93, 0], [35, 255, 255])

    green = color_filter(image, hsv_image, [25, 52, 72], [102, 255, 255])

    blue = color_filter(image, hsv_image, [94, 80, 2], [126, 255, 255])

    purple = color_filter(image, hsv_image, [133, 43, 46], [155, 255, 255])

    black = color_filter(image, hsv_image, [0, 0, 0], [179, 100, 130])

    white = color_filter(image, hsv_image, [0,0,168], [172,111,255])

    gray = color_filter(image, hsv_image, [0, 0, 168], [172, 111, 255])

    '''
    cv2.imshow("red", red)
    cv2.imshow("orange", orange)
    cv2.imshow("yellow", yellow)
    cv2.imshow("green", green)
    cv2.imshow("blue", blue)
    cv2.imshow("purple", purple)
    cv2.imshow("black", black)
    cv2.imshow("white", white)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    '''


    max_color_index = np.argmax(np.array([red, orange, yellow, green, blue, purple, black, white, gray]))

    return color_list[max_color_index]


def check(image):
    global dnn_yolo # for detect human
    global chair_det # for detect chairs
    global _frame

    min_x = 999999
    ans = -1
    cnt = 0 # how many chairs there are

    
    detections_2 = dnn_yolo.forward(image)[0]["det"]
    

    chair_1 = None
    chair_2 = None
    h,w,c = image.shape
    
    time.sleep(0.5)
    while chair_1 is None:
        time.sleep(0.1)
        image = _frame.copy()
        detections_1 = chair_det.forward(image)[0]["det"]
        # check chairs
        for i, detection in enumerate(detections_1):
            x1, y1, x2, y2, score, class_id = map(int, detection)
            # if class_id != 56: continue
            if cnt == 0:
                cnt += 1
                chair_1 = [x1, y1, x2, y2, score, class_id]
            else:
                chair_2 = [x1, y1, x2, y2, score, class_id]
                break
    
    # check people
    for i, detection in enumerate(detections_2):
        x1, y1, x2, y2, score, class_id = map(int, detection)
        # d = depth[(y1+y2)//2][(x1+x2)//2]
        if class_id == 0: # if there is people in the image
            if chair_1[0]-10 < (x1+x2)//2 < chair_1[2]+10: # if the person is in the range of chair 1
                if chair_2 is None:
                    return (chair_1[0]+chair_1[2])//2 , [chair_1[0], chair_1[1], chair_1[2], chair_1[3]]
                else:
                    return (chair_2[0]+chair_2[2])//2 , [chair_2[0], chair_2[1], chair_2[2], chair_2[3]]
            else: # if not, return the position of chair 1
                return (chair_1[0]+chair_1[2])//2 , [chair_1[0], chair_1[1], chair_1[2], chair_1[3]]
    if chair_2 is None: return (chair_1[0]+chair_1[2])//2, [chair_1[0], chair_1[1], chair_1[2], chair_1[3]]
    if chair_1[0] < chair_2[0]:
        return (chair_1[0]+chair_1[2])//2 , [chair_1[0], chair_1[1], chair_1[2], chair_1[3]]
    else:
        return (chair_2[0]+chair_2[2])//2 , [chair_2[0], chair_2[1], chair_2[2], chair_2[3]]
    
    
    
if __name__ == "__main__":
    rospy.init_node("task3")
    rospy.loginfo("task3 started!")

    _frame = None
    rospy.Subscriber("/cam2/color/image_raw", Image, callback_image)
    _depth = None
    rospy.Subscriber("/cam2/depth/image_raw", Image, callback_depth)
    print("camera is ready")

    _voice = None
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    _pub_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    
    print("speaker is ready")
    
    chassis = RobotChassis()
    
    msg_cmd = Twist()
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    
    clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
    
    print("robot chassis is ready")
    
    dnn_pose = HumanPoseEstimation(device_name="GPU")

    imu = None
    rospy.Subscriber("/imu/data", Imu, imu_callback)

    # location
    task = {"door":(-0.435, 4.16, 0.15), "host":(-1.32, -0.693, -1.57), "seats":(-0.428, -0.0327, 0.09), "start":(-2.8, -4.55, 3.14), "wait_1":(-1.94, 2.78, 3.14)}

    step = 1
    name = None
    drink = None
    cnt = 0 # how many times the whole flow has executed
    guest = [['', ''],['', '']]
    guest_char = []
    detect_seat = False

    dnn_yolo = Yolov8(device_name="GPU")
    # dnn_gender = AgeGenderRecognition(device_name="GPU")

    chair_det = Yolov8("chair_V2", device_name="GPU")
    chair_det.classes = ["chair"]
    
    glasses_det = Yolov8("best_V1", device_name="GPU")
    glasses_det.classes = ["glasses"]
    
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    
    print("all models are loaded")

    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        print('hi')
        if _frame is None: continue
        if step == 1:
            # chassis.move_to(*task["start"])
            # detect a guest
            h, w, c = _frame.shape
            image = _frame.copy()[:,w//3:w//3*2,:]
            poses = dnn_pose.forward(image)
            cv2.imshow("frame", image)
            key = cv2.waitKey(1)
            if key in [ord('q'), ord(' ')]:
                break
            if len(poses) == 0:continue
            print("human detected")
            # chassis.move_to(*task["door"])
            
            step += 1
        elif step == 2:
            # get name and drink
            
            if cnt == 0:
                image = _frame.copy()
                
                t1 = "what is the gender of the person in the picture?"
                # t2 = "what is the upper cloth's color of the person in the picture?"
                # t3 = "what is the pants' color of the person in the picture?"
                # t4 = "Is the person in the picrure wearing glasses?"
                t5 = "Is the person in the picture wearing a hat?"
                

                

                # prepare inputs
                encoding1 = processor(image, t1, return_tensors="pt")
                # encoding2 = processor(image, t2, return_tensors="pt")
                # encoding3 = processor(image, t3, return_tensors="pt")
                # encoding4 = processor(image, t4, return_tensors="pt")
                encoding5 = processor(image, t5, return_tensors="pt")

                # forward pass
                outputs1 = model(**encoding1)
                # outputs2 = model(**encoding2)
                # outputs3 = model(**encoding3)
                # outputs4 = model(**encoding4)
                outputs5 = model(**encoding5)
                
                logits1 = outputs1.logits
                # logits2 = outputs2.logits
                # logits3 = outputs3.logits
                # logits4 = outputs4.logits
                logits5 = outputs5.logits
                
                idx1 = logits1.argmax(-1).item()
                # idx2 = logits2.argmax(-1).item()
                # idx3 = logits3.argmax(-1).item()
                # idx4 = logits4.argmax(-1).item()
                idx5 = logits5.argmax(-1).item()

                guest_char.append(model.config.id2label[idx1]) # gender
                
                

                # guest_char.append(model.config.id2label[idx2])
                # guest_char.append(model.config.id2label[idx3])
                # check color
                image_1 = body_slice(image, 5, 12) # 5:shoulder_L 12:hip_R
                image_2 = body_slice(image, 11, 14) # 14: knee_R
                print(image_1)
                
                upper_color = color_detect(image_1)
                lower_color = color_detect(image_2)
                guest_char.append(upper_color)
                guest_char.append(lower_color)
                glasses_detection = glasses_det.forward(image)[0]["det"]
                if glasses_detection is None:
                    guest_char.append("no")
                else:
                    guest_char.append("yes")
                # guest_char.append(model.config.id2label[idx4]) # glasses
                guest_char.append(model.config.id2label[idx5]) # hat

                print("Predicted answer:", model.config.id2label[idx1])
                rospy.logerr(f"upper color: {upper_color}")
                rospy.logerr(f"lower color: {lower_color}")
                
                # print("Predicted answer:", model.config.id2label[idx2])
                # print("Predicted answer:", model.config.id2label[idx3])
                # print("Predicted answer:", model.config.id2label[idx4])
                print("Predicted answer:", model.config.id2label[idx5])
                
                print("---------------The first guest's features:--------------------")
                print(f"This guest is a {model.config.id2label[idx1]}")
                print(f"This guest is wearing a {upper_color} cloth")
                print(f"This guest is wearing a {lower_color} pants")
                if glasses_detection is None: print(f"The guest is not wearing a glasses")
                else: print(f"The guest is wearing a glasses")
                print("---------------Features Description end-----------------------")

                
                '''
                # check color
                poses = dnn_pose.forward(image)
                P = []
                for pose in poses:
                    for i, p in enumerate(pose):
                        x, y, c = map(int, p)
                        if i in [5,11,13]:
                            P.append([x,y])
                
                upper_color = detect_color(image[P[0][1]:P[1][1], P[0][0]:P[1][0], :])
                down_color = detect_color(image[P[1][1]:P[2][1], P[1][0]:P[2][0], :])
                guest_char.append(upper_color)
                guest_char.append(down_color)
                '''
                # check gender
                # age, gender = dnn_gender.forward(image)
                # gender = dnn_gender.genders_label[gender]
                # rospy.logwarn(gender)
                # guest_char.append(gender)
            rospy.loginfo("characteristics are recorded!")
            say("Welcome to the party. I am a domestic robot, what is your name? Please say it louder and use complete sentence. For example, my name is Olivia")
            
            while name is None:
                while _voice is None or "name" not in _voice.text: time.sleep(0.05)
                name = text_to_ans(''.join(_voice.text.split(' ')), "name") # use text_to_ans
                if name is None:
                    time.sleep(0.1)
                    say("Sorry, could you please repeat it?")
                    time.sleep(4)
            print(name)
            
            say("Nice to meet you. May I know your favorite drink? Please use complete sentence. For example, my favorite drink is soda.")
            while drink is None:
                while _voice is None or "drink" not in _voice.text: time.sleep(0.05)
                drink = text_to_ans(''.join(_voice.text.split(' ')), "drink") # use text_to_ans
                if drink is None: say("Sorry, could you please repeat it?")
                time.sleep(4)
            print(drink)
            guest[cnt][0], guest[cnt][1] = name, drink
            say("Ok. Please stand behind me and follow me")
            name, drink = None, None
            _voice = None
            step += 1
        elif step == 3:
            
            clear_costmaps
            chassis.move_to(*task["wait_1"])
            clear_costmaps
            chassis.move_to(*task["host"])
            
            rospy.loginfo("The robot has arrived at the front of the host")
            arm_control(1.57,0,0,0)
            say(f"Dear master, this is {guest[cnt][0]}. {guest[cnt][0]} wants to order a cup of {guest[cnt][1]}")
            say("I will guide the guest to sit now")
            arm_control(0.002, -1.049, 0.357, 0.703)
            clear_costmaps
            
            chassis.move_to(*task["seats"])
            rospy.loginfo("the robot has arrived at the front of the seats")
            step += 1
        elif step == 4:
            if _frame is None: continue
            rospy.logerr("step 4 started")
            image = _frame.copy()
            detections = dnn_yolo.forward(image)[0]["det"]
            if detections is None: continue
            # depth = _depth.copy()
            cx, pos = check(image) # determine which chair should the robot offer to the guest
            rospy.logwarn("the position is obtained")
            cv2.rectangle(image, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), 5)
            if not detect_seat:
                detect_seat = True
                
                cv2.imwrite(f"/home/pcms/Desktop/detected_seat_{cnt}.jpg", image)
            
            if 310 < cx < 330:
                print("-----------------angular reached-------------------")
                image_seat = _frame.copy()
                cv2.rectangle(image_seat, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), 5)
                cv2.imwrite(f"/home/pcms/Desktop/RCJ_seat{cnt}.jpg", image_seat)
                step += 1
                msg_cmd.angular.z = 0
            else:
                vel = angular_PID(cx, 320)
                msg_cmd.angular.z = vel
                print(f"velocity: {vel}")
            pub_cmd.publish(msg_cmd)
            cv2.imshow("frame", image)
            cv2.waitKey(1)
        elif step == 5:
            say("Please have a seat in front of me")
            time.sleep(5)
            
            if cnt == 1:
                say(f"The person who sits next to you is {guest[0][0]}. {guest[0][0]} is a {guest_char[0]}. {guest[0][0]} wears a cloth in {guest_char[1]} and a pair of pants in {guest_char[2]}") # To be edited
                if guest_char[3] == "yes":
                    say(f"{guest[0][0]} is wearing glasses")
            rospy.sleep(1.5)
            say("Please have a seat, and I will leave now")
            clear_costmaps
            chassis.move_to(*task["wait_1"])
            clear_costmaps
            chassis.move_to(*task["door"])
            
            cnt += 1
            detect_seat = False
            step = 1
            name = None
            drink = None
            if cnt == 2:
                break
    rospy.loginfo("task3 end!")
