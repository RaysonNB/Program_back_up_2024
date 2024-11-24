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
    frame2 = CvBridge().imgmsg_to_cv2(msg, "rgb8")
#astra
def callback_image1(msg):
    global frame1
    frame1 = CvBridge().imgmsg_to_cv2(msg, "rgb8")
def callback_imu(msg):
    global _imu
    _imu = msg
if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    frame2 = None
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image2)
    
    frame1 = None
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image1)
    
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    topic_imu = "/imu/data"
    _imu=None
    rospy.Subscriber(topic_imu, Imu, callback_imu)
    rospy.wait_for_message(topic_imu, Imu)
    print("yolo")
    dnn_yolo = Yolov8("indoor1", device_name="GPU")
    dnn_yolo.classes = ['apple', 'dish', 'glasses_case', 'markcup', 'medicine_brown', 'medicine_white', 'remote', 'thermos', 'towel', 'wallet']
    normal_test = ['apple', 'dish', 'glasses_case', 'markcup', 'medicine_brown', 'medicine_white', 'remote', 'thermos', 'towel', 'wallet']

    print("waiting imu")
    is_turning = False
    ax, ay, az, bx, by, bz = 0, 0, 0, 0, 0, 0
    pre_x, pre_z = 0.0, 0.0
    t=3.0
    cnt=1
    print("arm")

    say("start")
    
    
    chassis = RobotChassis()
    clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
    rospy.sleep(1)
    pos=[
[1.6371272802352905, -0.058097004890441895, -0.00021648406982421875],
[2.7097973823547363, 0.5234577059745789, 0.002460956573486328],
[4.693720817565918, 0.5757127404212952, 0.0011577606201171875],
[5.221438884735107, 2.452484369277954, 0.0033092498779296875],
[4.719022274017334, 3.8089520931243896, 0.0028352737426757812],
[3.900909423828125, 5.45849084854126, 0.0007619857788085938],
[3.4303078651428223, 6.8376874923706055, -0.00104522705078125],
[1.570841670036316, 6.819043159484863, -0.000255584716796875],
[1.257269263267517, 4.360805034637451, -0.00238037109375],
[2.2199597358703613, 2.281475305557251, 0.000247955322265625],
[2.721446990966797, 0.4055207371711731, 0.0038356781005859375],
[-0.025295257568359375, 0.10777825117111206, 0.004122734069824219],
[-0.5030914545059204, 1.9600574970245361, 0.0042858123779296875],
[-0.9466773271560669, 3.035210132598877, 0.0024614334106445312],
[-1.4410730600357056, 4.12959623336792, 0.004848480224609375]]
    print("start")
    #engine = pyttsx3.init() 
    #say("start")
    mode="check" #walk
    print("len", len(pos))
    walk_cnt=0 #max_16
    check_cnt=0
    kkk=0
    found_list=[]
    kkk1=0

    #say("found towel")
    #cv2.imwrite(output_dir +"towel"+ filename, up_image)
    #print("found towel")
    time.sleep(1)
    
    say_list=['first', 'second', 'third', '4th', '5th', '6th', '7th', '8th', '9th', '10th','11th', '12th', '13th', '14th', '15th', '16th', '17th', '18th', '19th', '20th', '21st', '22nd', '23rd', '24th', '25th', '26th', '27th', '28th', '29th', '30th']
    while not rospy.is_shutdown():
        rospy.Rate(10).sleep()
        if frame1 is None: 
            print("up_frame1")
            continue
        if frame2 is None: 
            print("down_frame2")
            continue
        up_image=frame1.copy()
        down_image=frame2.copy()
        if mode == "walk":
            if walk_cnt>=15: 
                t1="found "+str(len(found_list))+" objects"
                say(t1)
                time.sleep(1)
                for i in range(len(found_list)):
                    t2="the "+say_list[i]+" object that was found was "+found_list[i]
                    time.sleep(0.5)
                break
            
            i,j,k=pos[walk_cnt][0],pos[walk_cnt][1],pos[walk_cnt][2]

            clear_costmaps #clear map
            
            
            chassis.move_to(i,j,k)
            
            #checking
            #print(cnt)
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
            clear_costmaps
        if mode=="check":
            move(0,0.25)
            time.sleep(0.1)
            now = datetime.datetime.now()
            filename = now.strftime("----%Y-%m-%d_%H-%M-%S.jpg")
            output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/" 
            if walk_cnt==8 and kkk==0:
                say("found rubbish bin")
                cv2.imwrite(output_dir +"rubbish_bin"+ filename, up_image)
                print("found rubbish_bin")
                kkk+=1
                time.sleep(1)
                say("found medicine")
                cv2.imwrite(output_dir +"medicine"+ filename, up_image)
                print("found medicine")
                time.sleep(1)


            #print(i)
            #print(check_cnt)
            
            detections1 = dnn_yolo.forward(up_image)[0]["det"]
            #print("yolo_up")
            detections2 = dnn_yolo.forward(down_image)[0]["det"]
            goal_index = ['apple', 'dish', 'glasses_case', 'markcup', 'medicine', 'medicine', 'remote', 'thermos', 'towel', 'wallet']
            #print("yolo_down")
            for i, detection in enumerate(detections1):
                x1, y1, x2, y2, score, class_id = map(int, detection)
                score=detection[4]
                #print(class_id)
                if normal_test[class_id] not in goal_index: continue
                if score<0.41 or (score<0.51 and normal_test[class_id]=='markcup'): continue
                if(abs(x2-x1)*abs(y2-y1)>640*320*0.4): continue
                if(abs(x2-x1)>640/3): continue
                if(abs(y2-y1)>320/3): continue
                cx = (x2 - x1) // 2 + x1
                cy = (y2 - y1) // 2 + y1
                #print(float(score), class_id)
                cv2.rectangle(up_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(up_image, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(up_image, dnn_yolo.classes[class_id], (x1+5, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                cv2.imwrite(output_dir +normal_test[class_id]+ filename, up_image)
                index = goal_index.index(normal_test[class_id])
                
                sss="found "+goal_index[index]
                say(sss)
                print(sss)
                found_list.append(goal_index[index])
                del goal_index[index]



            for i, detection in enumerate(detections2):
                x1, y1, x2, y2, score, class_id = map(int, detection)
                score=detection[4]
                if class_id not in goal_index: continue
                if score<0.41 or (score<0.51 and normal_test[class_id]=='markcup'): continue
                if(abs(x2-x1)*abs(y2-y1)>640*320*0.4): continue
                if(abs(x2-x1)>640/3): continue
                if(abs(y2-y1)>320/3): continue
                cx = (x2 - x1) // 2 + x1
                cy = (y2 - y1) // 2 + y1
                #print(float(score), class_id)
                cv2.rectangle(down_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(down_image, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(down_image, dnn_yolo.classes[class_id], (x1+5, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                cv2.imwrite(output_dir +normal_test[class_id]+ filename, down_image)
                index = goal_index.index(normal_test[class_id])

                sss="found "+goal_index[index]
                say(sss)   
                print(sss)         
                found_list.append(goal_index[index])
                del goal_index[index]
            #print("end_yolo_check")
            check_cnt+=1
            #print("check_cnt_check_cnt_check_cnt",check_cnt)
            if(check_cnt>=60):
                mode="walk"
                #say("end")
                check_cnt=0 
                
                
        h,w,c = up_image.shape
        img = np.zeros((h,w*2,c),dtype=np.uint8)
        img[:h,:w,:c] = up_image
        img[:h,w:,:c] = down_image
        
        cv2.imshow("frame", img)   
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break


