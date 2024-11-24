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

"""
To-do list:
- Train chair's model
- Change the part of reporting names and drinks to host
- Look in the direction while navigation ***到時候再改***
- Look at the person being described ***Finish coding, waiting to be tested***
"""
def angular_PID(cx, tx):
    e = tx - cx
    p = 0.0015
    z = p * e
    if z > 0:
        z = min(z, 0.2)
        z = max(z, 0.05)
    if z < 0:
        z = max(z, -0.2)
        z = min(z, -0.05)
    return z


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
    name_list = ["Sophie", "Fleur", "Kevin", "Julia", "Gabrielle", "Jesse", "Emma", "Robin", "Noah", "Sara", "John", "Harrie", "Laura", "William","Liam", "Peter", "Hayley","Lucas","Susan"]
    possibility_name = [["Sophie", "selfie"], ["Fleur", "villa", "fuller", "floor", "flood","lure"], ["Kevin"], ["Julia"], ["Gabrielle", "Gabriella"], ["Jesse", "Jessie", "Jessa"], ["Emma"], ["Robin", "Ruben", "moving", "Reuben"], ["Noah"], ["Sara", "Sarah"], ["John"], ["Harrie", "Harry","Harvey", "hardly"], ["Laura","Lara"], ["William","volume","villain","radium"],["Liam", "lion", "Leanne"], ["Peter", "Peeta", "pizza", "pita"], ["Hayley"],["Lucas"],["Susan"]]

    drink_list = ["Espresso", "Pepsi","Milk","Coffee","Orange juice", "strawberry juice", "tea", "lemon juice", "apple juice"]
    possibility_drink = [["Espresso", "expiration"], ["Pepsi"],["Milk",'moke'],["Coffee","corky"],["orangejuice", "oranges", "orange"], ["strawberryjuice", "littlebetterjuice"], ["tea"], ["lemonjuice"], ["applejuice","approaches"]]

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
    '''
    if p1 + p2 == -2: # detect the lower part of the body
        x1, y1, x2, y2 = int(poses[idx][5][0]), int(poses[idx][12][1]), int(poses[idx][12][0]), 479
        '''
    x1, x2 = min(x1,x2), max(x1,x2)
    y1, y2 = min(y1, y2), max(y1,y2)
    print(f"{x1=} {x2=} {y1=} {y2=}")
    return image[y1:y2, x1:x2, :].copy(), [x1,x2,y1,y2]
    
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
    print(hsv_image[:,:,2])
    with open("/home/pcms/Desktop/color.txt", 'a') as f:
        f.write(f"{hsv_image[:,:,2]}")
    color_list = ["red", "orange", "yellow", "green", "blue", "purple", "black", "white", "gray"]

    # find the pixels that refer to the specific color
    red = color_filter(image, hsv_image, [120, 254, 200], [179, 255, 255])

    orange = color_filter(image, hsv_image, [5, 75, 0], [21, 255, 255])

    yellow = color_filter(image, hsv_image, [22, 93, 0], [35, 255, 255])

    green = color_filter(image, hsv_image, [25, 52, 72], [102, 255, 255])

    blue = color_filter(image, hsv_image, [94, 80, 2], [126, 255, 255])

    purple = color_filter(image, hsv_image, [133, 43, 46], [155, 255, 255])

    black = color_filter(image, hsv_image, [0, 0, 0], [179, 100, 60])

    white = color_filter(image, hsv_image, [0,0,135], [172,111,255])

    gray = color_filter(image, hsv_image, [0, 0, 61], [172, 111, 134])


    max_color_index = np.argmax(np.array([red, orange, yellow, green, blue, purple, black, white, gray]))
    print(f"{black=}")
    print(f"{white=}")
    print(f"{gray=}")
    return color_list[max_color_index]


# Not yet tested
def check(image):
    global chair_det # for detect chairs
    global _frame, _depth

    min_x = 999999
    ans = -1
    cnt = 0 # how many chairs there are

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = chair_det.forward(image)[0]["det"]
    # detections = dnn_yolo.forward(image)[0]["det"]
    

    # chair_1 = None
    # chair_2 = None
    h,w,c = image.shape
    max_x = 0
    r1, r2 = -1, []
    for i, detection in enumerate(detections):
        print(detection)
        x1, y1, x2, y2, score, class_id = map(int, detection)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        if detection[4] > 0.7 and x1 > max_x:
            max_x, r1, r2 = x1, (x1+x2)//2, [x1, y1, x2, y2]
    return r1, r2
    
    
def glasses(image):
    global glasses_det
    n_glasses = glasses_det.forward(image)[0]["det"]
    for x1,y1,x2,y2,score,class_id in n_glasses:
        if score > 0.65:
            return True
    return False
    

def human_detection(image):
    global dnn_pose
    poses = dnn_pose.forward(image)
    n_joints = 5
    for pose in poses:
        if len(pose) > n_joints:
            return True
    return False
            
def check_human(image):
    global dnn_yolo, _depth
    detections = dnn_yolo.forward(image)[0]["det"]
    for detection in detections:
        x1,y1,x2,y2,score,class_id = map(int, detection)
        if class_id != 0: continue
        cx, cy = (x1+x2)//2, (y1+y2)//2
        d = _depth[cy][cx]
        if d < 3000 and detection[4] > 0.75:
            return cx
    return -1
    
def turn(angle: float):
    q = [
        imu.orientation.x,
        imu.orientation.y,
        imu.orientation.z,
        imu.orientation.w
    ]
    roll, pitch, yaw = euler_from_quaternion(q)
    target = yaw + angle
    if target > np.pi:
        target = target - np.pi * 2
    elif target < -np.pi:
        target = target + np.pi * 2
    turn_to(target, 0.4)
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
    task = {"door":(3.011, -7.097, -0.012), "host":(-1.32, -0.693, -1.57), "seats":(0.142, -4.907, 0), "start":(-2.8, -4.55, 3.14), "wait_1":(1.97, 1.73, -0.005)}

    step = 1
    name = None
    drink = None
    cnt = 0 # how many times the whole flow has executed
    guest = [['', ''],['', '']]
    guest_char = []
    detect_seat = False

    dnn_yolo = Yolov8(device_name="GPU")
    # dnn_gender = AgeGenderRecognition(device_name="GPU")

    chair_det = Yolov8("chair_rayson", device_name="GPU")
    chair_det.classes = ["chair"]
    # chair_det = Yolov8("chair_V2", device_name="GPU")
    # chair_det.classes = ["chair"]
    
    glasses_det = Yolov8("best_V1", device_name="GPU")
    glasses_det.classes = ["glasses"]
    
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    
    print("all models are loaded")

    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        if _frame is None:
            print("the camera failed")
            continue
        if step == 1:
            
            print("step 1")
            ## chassis.move_to(*task["start"]) hh 
            # detect a guest
            h, w, c = _frame.shape
            image = _frame.copy()[:,w//3:w//3*2,:]
            poses = dnn_pose.forward(image)
            if not human_detection(image):continue
            image = dnn_pose.draw_poses(image,poses,0.1)
            cv2.imwrite("/home/pcms/Desktop/guest_detect.jpg", image)
            print("human detected")
            ## chassis.move_to(*task["door"])
            
            step += 1
        elif step == 2:
            # get name and drink
            
            if cnt == 0:
                time.sleep(2)
                image = _frame.copy()
                frame = _frame.copy()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite("/home/pcms/Desktop/feature_detection.jpg", image)
                t1 = "what is the gender of the person in the picture?"
                t2 = "Is the person wearing a mask?"
                t5 = "Is the person in the picture wearing a hat?"
                t3 = "Is the person a teenager, young adult, middle-aged person, or an old man?"
                t4 = "what is the color of the person's hair?"
                t6 = "what is the color of the person's cloth?"
                

                # prepare inputs
                encoding1 = processor(image, t1, return_tensors="pt")
                encoding2 = processor(image, t2, return_tensors="pt")
                encoding5 = processor(image, t5, return_tensors="pt")
                encoding3 = processor(image, t3, return_tensors="pt")
                encoding4 = processor(image, t4, return_tensors="pt")
                encoding6 = processor(image, t6, return_tensors="pt")

                # forward pass
                outputs1 = model(**encoding1)
                outputs2 = model(**encoding2)
                outputs5 = model(**encoding5)
                outputs3 = model(**encoding3)
                outputs4 = model(**encoding4)
                outputs6 = model(**encoding6)
                
                logits1 = outputs1.logits
                logits2 = outputs2.logits
                logits3 = outputs3.logits
                logits5 = outputs5.logits
                logits4 = outputs4.logits
                logits6 = outputs6.logits
                
                idx1 = logits1.argmax(-1).item()
                idx2 = logits2.argmax(-1).item()
                idx3 = logits3.argmax(-1).item()
                idx5 = logits5.argmax(-1).item()
                idx4 = logits4.argmax(-1).item()
                idx6 = logits6.argmax(-1).item()
                
                
                feature_gender = model.config.id2label[idx1] # gender
                feature_mask = model.config.id2label[idx2] # mask
                feature_age = model.config.id2label[idx3] #age
                feature_hair = model.config.id2label[idx4] # hair
                feature_cloth = model.config.id2label[idx6]
                
                guest_char.append(feature_gender) # gender
                
                
                
                image = frame.copy()
                # check color
                #image_1, List1 = body_slice(image, 5, 12) # 5:shoulder_L 12:hip_R
                #image_2, List2 = body_slice(image, 11, 14) # 14: knee_R
                #cv2.imwrite(f"/home/pcms/Desktop/upper.jpg", image_1)
                
                #upper_color = color_detect(image_1)
                
                guest_char.append(feature_cloth)
                glasses_detection = glasses(image)
                
                if not glasses_detection:
                    guest_char.append("no")
                else:
                    guest_char.append("yes")

                guest_char.append(model.config.id2label[idx5]) # hat

                print("Predicted answer:", model.config.id2label[idx1])
                
                print("Predicted answer:", model.config.id2label[idx5])
                
                print("---------------The first guest's features:--------------------")
                print(f"This guest is a {model.config.id2label[idx1]}")
                if glasses_detection == "no": print(f"The guest is not wearing a glasses")
                else: print(f"The guest is wearing a glasses")
                print(f"This guest is a {feature_age}")
                print(f"The hair color of the guest is {feature_hair}")
                print(f"The cloth color of the guest is {feature_cloth}.")
                print("---------------Features Description end-----------------------")

            rospy.loginfo("characteristics are recorded!")
            say("Welcome to the party. I am a domestic robot, what is your name? Please say it louder and use complete sentence. For example, my name is Olivia")
            
            rospy.wait_for_message("/voice/text", Voice)
            time.sleep(0.1)
            name = text_to_ans(''.join(_voice.text.split(' ')), "name") # use text_to_ans
            _voice=None
            while name is None: 
                say("sorry, could you please repeat")
                rospy.wait_for_message("/voice/text", Voice)
                time.sleep(0.1)
                name = text_to_ans(''.join(_voice.text.split(' ')), "name") # use text_to_ans
            _voice=None
            print(name)
            
            say("Nice to meet you. May I know your favorite drink? Please use complete sentence. For example, my favorite drink is soda.")
            rospy.wait_for_message("/voice/text", Voice)
            drink = text_to_ans(''.join(_voice.text.split(' ')), "drink") # use text_to_ans
            while drink is None: 
                say("sorry, could you please repeat")
                rospy.wait_for_message("/voice/text", Voice)
                time.sleep(0.1)
                drink = text_to_ans(''.join(_voice.text.split(' ')), "drink") # use text_to_ans
            time.sleep(2)
            print(drink)
            guest[cnt][0], guest[cnt][1] = name, drink
            say("Ok. Please stand behind me and follow me")
            name, drink = None, None
            _voice = None
            clear_costmaps
            chassis.move_to(*task["seats"])
            step += 1
        elif step == 3:
            
            #chassis.move_to(*task["wait_1"])
            
            
            rospy.loginfo("The robot has arrived at the front of the chairs")
            if cnt == 0:
                cx = check_human(_frame)
                cv2.imshow("frame", _frame)
                cv2.waitKey(1)
                if cx == -1:
                    msg_cmd.angular.z = 0.2
                    pub_cmd.publish(msg_cmd)
                    continue
                else:    
                    msg_cmd.angular.z = 0.0
                    pub_cmd.publish(msg_cmd)
                    
                if 305 < cx < 335:
                    msg_cmd.angular.z = 0.0
                    pub_cmd.publish(msg_cmd)
                else:
                    msg_cmd.angular.z = 0.2
                    pub_cmd.publish(msg_cmd)
                    continue
                say("Dear master, I want to introduce the first guest to you.")
                time.sleep(2)
                say("Dear guest, could you please stand behind me and stay there please? Thank you so much.")
                time.sleep(2)
                turn(3.1415926)
            # arm_control(1.57,0,0,0)
            clear_costmaps
            # chassis.move_to(task["chairs"][0], task["chairs"][1], task["chairs"][2]+3.14)
            
            say(f"This is {guest[cnt][0]}. {guest[cnt][0]} wants to order a cup of {guest[cnt][1]}")
            # say("I will guide the guest to sit now")
            # arm_control(0.002, -1.049, 0.357, 0.703)
            
            rospy.loginfo("the robot has finished describing the guest")
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
            if len(pos) == 0:
                print("no chairs are found")
                msg_cmd.angular.z = 0.2
                pub_cmd.publish(msg_cmd)
                continue
            cv2.rectangle(image, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), 5)
            print("break try loop")
            msg_cmd.angular.z = 0.0
            pub_cmd.publish(msg_cmd)
            
            if 305 < cx < 335:
                print("-----------------angular reached-------------------")
                image_seat = _frame.copy()
                cv2.rectangle(image_seat, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), 5)
                cv2.imwrite(f"/home/pcms/Desktop/detected_seat{cnt}.jpg", image_seat)
                step += 1
                msg_cmd.angular.z = 0
            else:
                vel = angular_PID(cx, 320)
                msg_cmd.angular.z = vel
                print(f"velocity: {vel}")
            pub_cmd.publish(msg_cmd)
            cv2.imshow("frame", image)
            cv2.waitKey(1)
        elif step == 4:
            print("stop")
        elif step == 5:
            say("Please have a seat in front of me")
            time.sleep(5)
            
            if cnt == 1:
                say(f"The person who sits next to you is {guest[0][0]}. {guest[0][0]} is a {guest_char[0]}. {guest[0][0]} wears a cloth in {guest_char[1]}. {guest[0][0]} is a {feature_age} person.") # To be edited
                if glasses_detection:
                    say(f"{guest[0][0]} is wearing glasses")
                if feature_mask == 'yes':
                    say(f"{guest[0][0]} is wearing a mask")
                say(f"{guest[0][0]} has {feature_hair} hair")
            rospy.sleep(1.5)
            say("Please have a seat, and I will leave now")
            clear_costmaps
            #chassis.move_to(*task["wait_1"])
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
