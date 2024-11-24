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
import time
from sensor_msgs.msg import Imu
from std_srvs.srv import Empty
from RobotChassis import RobotChassis
import datetime
from mr_voice.msg import Voice
from std_msgs.msg import String
from gtts import gTTS
from playsound import playsound


def say(g):
    tts = gTTS(g)

    # Save the speech as an audio file
    speech_file = "speech.mp3"
    tts.save(speech_file)

    # Play the speech
    playsound(speech_file)
def callback_voice(msg):
    global s
    s = msg.text
def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)
#gemini2
def callback_image2(msg):
    global frame2
    frame2 = CvBridge().imgmsg_to_cv2(msg, "bgr8")
#astra
def callback_image1(msg):
    global frame1
    frame1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")
def callback_imu(msg):
    global _imu
    _imu = msg
    global engine
    engine.say(s) 
    engine.runAndWait() 
if __name__ == "__main__":    
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    frame2 = None
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image2)
    
    frame1 = None
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image1)
    '''
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    topic_imu = "/imu/data"
    _imu=None
    rospy.Subscriber(topic_imu, Imu, callback_imu)
    rospy.wait_for_message(topic_imu, Imu)'''
    print("yolo")
    dnn_yolo = Yolov8("best", device_name="GPU")
    dnn_yolo.classes = ['apple', 'dish', 'glasses_case', 'markcup', 'medicine_brown', 'medicine_white', 'remote', 'thermos', 'towel', 'wallet']
    normal_test = ['apple', 'dish', 'glasses_case', 'markcup', 'medicine_brown', 'medicine_white', 'remote', 'thermos', 'towel', 'wallet']

    print("waiting imu")
    is_turning = False
    ax, ay, az, bx, by, bz = 0, 0, 0, 0, 0, 0
    pre_x, pre_z = 0.0, 0.0
    t=3.0
    cnt=1
    print("arm")
    '''
    Kinda = np.loadtxt(RosPack().get_path("mr_dnn") + "/Kinda.csv")
    dnn_yolo = Yolov8("bagv4")
    dnn_follow = Yolov8("yolov8n")
    dnn_yolo.classes = ['obj']'''
    print("run")

    say("start")
    
    '''
    chassis = RobotChassis()
    clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)'''
    rospy.sleep(1)
    pos=[[2.4542484283447266, 1.324235439300537, 0.00226593017578125],
     [1.1773695945739746, 1.6855344772338867, 0.00010776519775390625],
     [0.1447134017944336, 1.4833669662475586, 0.0026798248291015625],
     [-1.2144417762756348, 1.562230110168457, 0.0029201507568359375],
     [-2.094003200531006, 1.9802799224853516, 0.005820274353027344],
     [-2.7831168174743652, 0.6641445159912109, 0.0009584426879882812],
     [-2.740118980407715, -0.789093017578125, 0.0042743682861328125],
     [-2.692660331726074, -2.3315939903259277, 0.0006628036499023438],
     [-2.4997949600219727, -3.9404232501983643, 0.004380226135253906],
     [-1.3480339050292969, -4.239188194274902, 0.0008668899536132812],
     [-0.43129730224609375, -4.575753211975098, 0.004513740539550781],
     [0.15888595581054688, -2.9458394050598145, 0.0028638839721679688],
     [-0.020209312438964844, -1.3719573020935059, -0.00025081634521484375],
     [-0.12377357482910156, 0.8423192501068115, 0.00299835205078125],
     [2.2681922912597656, 1.6104834079742432, 0.0028314590454101562],
     [2.797421455383301, 0.4407784938812256, 0.0015544891357421875],
        [2.4160289764404297,-0.7244741916656494,0],
        [2.3478660583496094,-2.331594467163086,0]]
    print("start")
    #engine = pyttsx3.init() 
    #say("start")
    mode="check"
    walk_cnt=0 #max_18
    check_cnt=0
    while not rospy.is_shutdown():
        rospy.Rate(10).sleep()
        if frame1 is None: 
            print("up_frame1")
            continue
        if frame2 is None: 
            print("down_frame2")
            continue
        up_image=frame2.copy()
        down_image=frame1.copy()
        '''
        if mode == "walk":
            i,j,k=pos[walk_cnt][0],pos[walk_cnt][1],pos[walk_cnt][2]

            clear_costmaps #clear map
            
            
            chassis.move_to(i,j,k)
            
            #checking
            print(cnt)
            cnt+=1
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
            mode="check"
            check_cnt=0
            walk_cnt+=1
                '''
        if mode=="check":
            #move(0,0.25)
            #time.sleep(0.1)
            #print(i)
            print(check_cnt)
            now = datetime.datetime.now()
            filename = now.strftime("----%Y-%m-%d_%H-%M-%S.jpg")
            output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/" 
            
            detections1 = dnn_yolo.forward(up_image)[0]["det"]
            print("yolo1")
            detections2 = dnn_yolo.forward(down_image)[0]["det"]
            goal_index = ['apple', 'dish', 'glasses_case', 'markcup', 'medicine_brown', 'medicine_white', 'remote', 'thermos', 'towel', 'wallet']
            print("yolo2")
            for i, detection in enumerate(detections1):
                x1, y1, x2, y2, score, class_id = map(int, detection)
                score=detection[4]
                print(class_id)
                if normal_test[class_id] not in goal_index: continue
                if score<0.4: continue


                cx = (x2 - x1) // 2 + x1
                cy = (y2 - y1) // 2 + y1
                print(float(score), class_id)
                cv2.rectangle(up_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(up_image, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(up_image, str(class_id), (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                cv2.imwrite(output_dir +normal_test[class_id]+ filename, up_image)
                index = goal_index.index(normal_test[class_id])
                sss="found "+goal_index[index]
                say(sss)
                del goal_index[index]



            for i, detection in enumerate(detections2):
                x1, y1, x2, y2, score, class_id = map(int, detection)
                score=detection[4]
                if class_id not in goal_index: continue
                if score<0.4: continue


                cx = (x2 - x1) // 2 + x1
                cy = (y2 - y1) // 2 + y1
                print(float(score), class_id)
                cv2.rectangle(down_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(down_image, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(down_image, str(class_id), (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                cv2.imwrite(output_dir +normal_test[class_id]+ filename, down_image)
                index = goal_index.index(normal_test[class_id])
                sss="found "+goal_index[index]
                say(sss)            
                del goal_index[index]
            print("end_yolo_check")
            check_cnt+=1
            if(check_cnt>=290):
                mode="walk"
                
                
        h,w,c = up_image.shape
        upout=cv2.line(up_image, (320,0), (320,500), (0,255,0), 5)
        downout=cv2.line(down_image, (320,0), (320,500), (0,255,0), 5)
        img = np.zeros((h,w*2,c),dtype=np.uint8)
        img[:h,:w,:c] = upout
        img[:h,w:,:c] = downout
        
        cv2.imshow("frame", img)   
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
