#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import random
import subprocess
from mr_voice.msg import Voice
from std_msgs.msg import String
import math
import time
from pcms.pytorch_models import *
from pcms.openvino_yolov8 import *
from pcms.openvino_models import Yolov8,HumanPoseEstimation
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import Imu

def say_cn(a):
    text = str(a)
    process = subprocess.Popen(['espeak-ng', '-v', 'yue', text])
    process.wait()


mleft = []
mright = []

def draw(event,x,y,flags,param):
    if(event == cv2.EVENT_LBUTTONDBLCLK and mleft == []):
        mleft.append(x)
        mleft.append(y)
    elif(event == cv2.EVENT_LBUTTONDBLCLK and mleft != [] and mright == []):
        mright.append(x)
        mright.append(y)
def get_distance(px,py,pz,ax,ay,az,bx,by,bz):
    A,B,C,p1,p2,p3,qx,qy,qz,distance=0,0,0,0,0,0,0,0,0,0
    A=int(bx)-int(ax)
    B=int(by)-int(ay)
    C=int(bz)-int(az)
    p1=int(A)*int(px)+int(B)*int(py)+int(C)*int(pz)
    p2=int(A)*int(ax)+int(B)*int(ay)+int(C)*int(az)
    p3=int(A)*int(A)+int(B)*int(B)+int(C)*int(C)
    #print("1",p1,p2,p3)
    if (p1-p2)!=0 and p3!=0:
        t=(int(p1)-int(p2))/int(p3)
        qx=int(A)*int(t) + int(ax)
        qy=int(B)*int(t) + int(ay)
        qz=int(C)*int(t) + int(az)
        return int(int(pow(((int(qx)-int(px))**2 +(int(qy)-int(py))**2+(int(qz)-int(pz))**2),0.5)))
    return 0
          
            
def callback_depth(msg):
    global depth
    tmp = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    depth = np.array(tmp, dtype=np.float32)


def dfs(x, y, statu):
    global depth_copy, depth_list, cnt
    if x < 1 or y < 1 or x > len(depth_copy[0]) - 2 or y > len(depth_copy) - 2:
        return
    if depth_copy[y][x] != 0:
        return
    depth_copy[y][x] = statu
    cnt += 1
    if x < 2:
        dfs(x + 1, y, statu)
        return
    if y < 2:
        dfs(x, y + 1, statu)
        return

    bx = False
    by = False
    if abs(abs(depth_list[y + 1][x] - depth_list[y][x]) - abs(depth_list[y][x] - depth_list[y - 1][x])) < 2:
        by = True
        dfs(x, y - 1, statu)
        dfs(x, y + 1, statu)
    if abs(abs(depth_list[y][x + 1] - depth_list[y][x]) - abs(depth_list[y][x] - depth_list[y][x - 1])) < 2:
        bx = True
        dfs(x + 1, y, statu)
        dfs(x - 1, y, statu)
    if not bx and not by:
        return
    return


def callback_image(msg):
    global image
    image = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def add_edge():
    for e in range(len(depth_list)):
        depth_list[e].insert(0, depth_list[e][0])  # 最左
        depth_list[e].insert(-1, depth_list[e][-1])  # 最右
    depth_list.insert(0, depth_list[0])
    depth_list.insert(-1, depth_list[-1])


def change_zero():
    for e in range(1, len(depth_list) - 1, 1):
        error = []
        for f in range(1, len(depth_list[e]) - 1, 1):
            if depth_list[e][f] == 0:
                if depth_list[e - 1][f] or depth_list[e - 1][f - 1] or depth_list[e - 1][f + 1] or depth_list[e + 1][f] or depth_list[e + 1][f - 1] or depth_list[e + 1][f + 1] or depth_list[e][f - 1] or depth_list[e][f + 1]:
                    for i in range(-1, 2, 1):
                        for j in range(-1, 2, 1):
                            if depth_list[e + i][f + j] != 0:
                                error.append(depth_list[e + i][f + j])
                    depth_list[e][f] = sum(error) // len(error)
def callback_voice(msg):
    global s
    s = msg.text
def say(a):
    global publisher_speaker
    publisher_speaker.publish(a)
    
def pose_draw(show):
    cx7,cy7,cx9,cy9,cx5,cy5,l,r=0,0,0,0,0,0,0,0
    s1,s2,s3,s4=0,0,0,0
    global ax,ay,az,bx,by,bz
    #for num in [7,9]: #[7,9] left, [8,10] right
    n1,n2,n3=6,8,10
    cx7, cy7 = get_pose_target(pose,n2)
    
    cx9, cy9 = get_pose_target(pose,n3)
    
    cx5, cy5 = get_pose_target(pose,n1)
    if cx7==-1 and cx9!=-1:
        s1,s2,s3,s4=cx5,cy5,cx9,cy9
        cv2.circle(show, (cx5,cy5), 5, (0, 255, 0), -1)
        ax,ay,az = get_real_xyz(depth2,cx5, cy5)
        _,_,l=get_real_xyz(depth2,cx5,cy5)
        cv2.circle(show, (cx9,cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(depth2,cx9, cy9)
        _,_,r=get_real_xyz(depth2,cx9,cy9)
    elif cx7 !=-1 and cx9 ==-1:
        s1,s2,s3,s4=cx5,cy5,cx7,cy7
        cv2.circle(show, (cx5,cy5), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(depth2,cx5, cy5)
        _,_,l=get_real_xyz(depth2,cx5,cy5)
        cv2.circle(show, (cx7,cy7), 5, (0, 255, 0), -1)
        bx,by,bz = get_real_xyz(depth2,cx7, cy7)
        _,_,r=get_real_xyz(depth2,cx7,cy7)
    elif cx7 ==-1 and cx9 == -1:
        pass
        #continue
    else:
        s1,s2,s3,s4=cx7,cy7,cx9,cy9
        cv2.circle(show, (cx7,cy7), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(depth2,cx7, cy7)
        _,_,l=get_real_xyz(depth2,cx7,cy7)
        cv2.circle(show, (cx9,cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(depth2,cx9, cy9)
        _,_,r=get_real_xyz(depth2,cx9,cy9)
    
    cv2.putText(show, str(int(l)), (s1 + 5, s2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    cv2.putText(show, str(int(r)), (s3 + 5, s4 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    return ax, ay, az, bx, by, bz
def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)
def get_pose_target(pose,num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])
    
    if len(p) == 0: return -1, -1
    return int(p[0][0]),int(p[0][1])    
def get_real_xyz(dp,x, y):
    a = 55.0 * np.pi / 180
    b = 86.0 * np.pi / 180
    d = dp[y][x]
    h, w = dp.shape[:2]
    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d
depth_copy = None
depth_list = []
xylist = []
color = {}
biggest_max = []

if __name__ == "__main__":
    rospy.init_node("BW_getdepth")
    rospy.loginfo("BW_getdepth start!")
    image = None
    depth = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth)
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image)
    rospy.wait_for_message("/camera/depth/image_raw", Image)
    rospy.wait_for_message("/camera/color/image_raw", Image)
    dnn_yolo = Yolov8("yolov8n", device_name="GPU")
    print("yolo")
    net_pose = HumanPoseEstimation(device_name="GPU")
    print("pose")
    isackcnt=0
    cimage = image.copy()
    xi = 150
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    
    print("speaker")
    step=1 #remember
    f_cnt=0
    step2="get" #remember
    A="dead"
    ax,ay,az,bx,by,bz=0,0,0,0,0,0
    pre_z, pre_x=0,0
    cur_z, cur_x=0,0
    test=0
    p_list=[]
    sb=0
    framecnt=0
    bottlecolor=["blue","orange","pink"]
    saidd=0
    get_b=0
    bottlecnt=0
    line_destory_cnt=0
    '''
    while True:
        k = cv2.waitKey(1)
        if k == -1:
            cv2.imshow("frame", image)
        else:
            cv2.destroyWindow('frame') 
            break'''
    depth_img = depth.copy()
    while not rospy.is_shutdown():
        rospy.Rate(50).sleep()
        sumd=0
        if step==1:
            line_frame=frame2.copy()
            frame2=frame2.copy()
            bottle=[]
            detections = dnn_yolo.forward(frame2)[0]["det"]
            #detections = dnn_yolo.forward(frame)[0]["det"]
            #print(detections)
            al=[]
            ind=0
            for i, detection in enumerate(detections):
                x1, y1, x2, y2, score, class_id = map(int, detection)
                score=detection[4]
                if class_id != 39: continue
                if score<0.3: continue
                al.append([x1, y1, x2, y2, score, class_id])
                #print(float(score), class_id)
                cv2.putText(frame2, str(class_id), (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            bb=sorted(al, key=(lambda x:x[0]))
            #print(bb)
            for i in bb:
                #print(i)
                x1, y1, x2, y2, _, _ = i
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 255), 4)
                cv2.putText(frame2, str(int(ind)), (cx,cy+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                ind+=1
                px,py,pz=get_real_xyz(depth2,cx, cy)
                cnt=get_distance(px,py,pz,ax,ay,az,bx,by,bz)
                cv2.circle(frame2, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(frame2, str(int(pz)), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)  
            if step2=="get":        
                outframe=frame2.copy()
                if A=="dead":
                    
                    t_pose=None
                    points=[]
                    poses = net_pose.forward(outframe)
                    
                    for i, pose in enumerate(poses):
                        point = []
                        for j, (x,y,preds) in enumerate(pose): #x: ipex 坐標 y: ipex 坐標 preds: 准度
                            if preds <= 0: continue
                            x,y = map(int,[x,y])
                            _,_,td=get_real_xyz(depth2,x, y)
                            if td>=2000: continue
                            if j in [8,10]:
                                point.append(j)
                        if len(point) == 2:
                            t_pose = poses[i]
                            break
                        #print(point)
                    
                    TTT=0
                    E=0
                    s_c=[]
                    
                    s_d=[]
                    ggg=0
                    flag=None
                    
                    if t_pose is not None:
                        ax, ay, az, bx, by, bz = pose_draw(outframe)
                        if len(bb) <3:
                            if bottlecnt>=3:
                                print("not enught bottle")
                                bottlecnt+=1
                            continue
                        for i, detection in enumerate(bb):
                            #print(detection)
                            x1, y1, x2, y2, score, class_id = map(int, detection)
                            score = detection[4]
                            #print(id)
                            if(class_id == 39):
                                ggg=1
                                bottle.append(detection)
                                E+=1
                                cx1 = (x2 - x1) // 2 + x1
                                cy1 = (y2 - y1) // 2 + y1
                                
                                
                                px,py,pz=get_real_xyz(depth2, cx1, cy1)
                                cnt=get_distance(px,py,pz,ax,ay,az,bx,by,bz)
                                
                                cnt=int(cnt)
                                if cnt!=0 and cnt<=600: cnt=int(cnt)
                                else: cnt=9999
                                s_c.append(cnt)
                                s_d.append(pz)
                                
                    if ggg==0: s_c=[9999]
                    TTT=min(s_c)
                    E=s_c.index(TTT)
                    for i, detection in enumerate(bottle):
                        #print("1")
                        x1, y1, x2, y2, score, class_id = map(int, detection)
                        if(class_id == 39):
                            if i == E and E!=9999 and TTT <=700:
                                cx1 = (x2 - x1) // 2 + x1
                                cy1 = (y2 - y1) // 2 + y1
                                print("hello")
                                cv2.putText(outframe, str(int(TTT)//10), (x1 + 5, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 0, 255), 2)
                                cv2.rectangle(outframe, (x1, y1), (x2, y2), (0, 0, 255), 5)
                                if i==0: b1+=1
                                if i==1: b2+=1
                                if i==2: b3+=1
                                
                                break
                                        
                            else:
                                v=s_c[i]
                                cv2.putText(outframe, str(int(v)), (x1+5, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if b1==max(b1,b2,b3): mark=0
                    if b2==max(b1,b2,b3): mark=1
                    if b3==max(b1,b2,b3): mark=2
                    if b1 >=10 or b2>=10 or b3>=10: 
                        A="turn"
                        gg=bb
                    #print("b1: %d b2: %d b3: %d" % (b1, b2, b3))
                if A=="turn":
                    if sb == 0:
                    
                        b1,b2,b3=0,0,0
                        if mark==0: say("the right bottle") #say("the right yellow bottle, which is the Vitamin C Sparkling Drink")
                        if mark==1: say("the middle bottle") #say("the middle green botlle, which is the only green tea here")
                        if mark==2: say("the left bottle") #say("the left blue bottle, which is the Oolong Tea")
                        
                        sb+=1
                    #if len(bb)<2: continue
                    #print(bb)
                    h,w,c = outframe.shape
                    print("mark",mark)
                    if mark==999 or len(bb)<(mark+1):
                        A="dead"
                        b1,b2,b3=0,0,0
                    if len(bb)!=3: continue
                    print(bb)
                    x1, y1, x2, y2, score, class_id = map(int, bb[mark])
                                
                        
                    if framecnt==0:
                        face_box = [x1, y1, x2, y2]
                        box_roi = outframe[face_box[1]:face_box[3] - 1, face_box[0]:face_box[2] - 1, :]
                        fh,fw=abs(x1-x2),abs(y1-y2)
                        box_roi=cv2.resize(box_roi, (fh*10,fw*10), interpolation=cv2.INTER_AREA)
                        #cv2.imshow("bottle", box_roi)  
                        get_b=mark
                        framecnt+=1
                    cx2 = (x2 - x1) // 2 + x1
                    cy2 = (y2 - y1) // 2 + y1
                    e = w//2-cx2
                    v = 0.0015 * e
                    if v > 0:
                        v = min(v, 0.3)
                    if v < 0:
                        v = max(v, -0.3)
                    move(0, v)
                    sumd+=v
                    if abs(e) <= 10:
                        time.sleep(1)
                        A="dead"
                        b1,b2,b3=0,0,0
                        mark=999
                        sb=0
                    
        i = 0
        tem_dlist = []
        tem_xylist = []
        if xi >= 600:
            break
        if isackcnt == 0:
            im2=image.copy()
            isackcnt+=1
        while True:
            h, w = depth_img.shape[:2]
            x, y = xi, h - 50 - i
            i += 10
            if i >= 330:
                depth_list.append(tem_dlist)
                xylist.append(tem_xylist)
                break
            d = depth_img[y][x]
            tem_xylist.append((x, y))
            tem_dlist.append(d)
            rospy.loginfo("%.2f" % d)
            gray = depth_img / np.max(depth_img)
            cv2.circle(im2, (x, y), 2, (0, 0, 255), 2)
            cv2.imshow("frame", im2)
            key_code = cv2.waitKey(1)
            if key_code in [ord('q'), 27]:
                break
        xi += 10
        
    add_edge()
    change_zero()
    
    for e in range(1, len(depth_list) - 1, 1):
        for f in range(1, len(depth_list[e]) - 1, 1):
            print(depth_list[e][f], end=" ")
        print()
        
    depth_copy = [[0 for e in range(len(depth_list[0]))] for f in range(len(depth_list))]
    
    for e in range(len(depth_list)):
        for f in range(len(depth_list[0])):
            depth_copy[e][f] = 0
        
    biggest = 0
    statue = 0
    cnt = 0
    for e in range(1, len(depth_list) - 1, 1):
        for f in range(1, len(depth_list[e]) - 1, 1):
            if depth_copy[e][f] == 0:
                cnt = 0
                statue += 1
                dfs(f, e, statue)
                biggest_max.append(cnt)
    for e in range(len(biggest_max)):
        if biggest_max[biggest] < biggest_max[e]:
            biggest = e
    print(f"the biggest flat is the flat {biggest}")
    
    for e in range(1,len(depth_copy)-1,1):
        for f in range(1,len(depth_copy[0])-1,1):
            error = [depth_copy[e+1][f],depth_copy[e-1][f],depth_copy[e][f+1],depth_copy[e][f-1]]
            check = error[0] == error[1] == error[2] == error[3]
            if depth_copy[e][f] not in error and check:
                depth_copy[e][f] = error[0]
                
    
    for e in range(1, statue + 1, 1):
        color[e] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    for e in range(1, len(depth_list) - 1, 1):
        for f in range(1, len(depth_list[e]) - 1, 1):
            print(depth_copy[e][f], end=' ')
        print()
    s = {}
    '''
    max_key = -1
    max_value = 0
    for e in range(1,len(depth_copy)-1,1):
        for f in range(1, len(depth_copy)-1, 1):
            if depth_copy[e][f] not in s:
                s[depth_copy[e][f]] = 1
            else:
                s[depth_copy[e][f]] += 1
            if s[depth_copy[e][f]] > max_value:
                max_value = s[depth_copy[e][f]]    
                max_key = depth_copy[e][f]
    '''
    image_copy = im2.copy()
    for e in range(1, len(xylist) + 1, 1):
        for f in range(1, len(xylist[0]) + 1, 1):
            circle_color = color[depth_copy[e][f]]
            #if depth_copy[e][f] == max_key:
            cv2.circle(image_copy, xylist[e-1][f-1], 2, circle_color, 2)
    cv2.namedWindow('result')
    cv2.setMouseCallback('result',draw)
    
    while True:
        k = cv2.waitKey(1)
        if mleft != [] and mright != []:
            image_copy = cv2.rectangle(image_copy, (mleft[0],mleft[1]), (mright[0],mright[1]), (0,0,255), 2)
            area = abs(mright[1] - mleft[1]) * abs(mright[0] - mright[0])
            print(mleft, mright)
            for e in range(abs(mleft[0] - 150) // 10 + 1, abs(mright[0] - 150)//10+1):
                for f in range(abs(430 - mright[1])//10+2,abs(430 - mleft[1])//10+2):
                    if depth_copy[e][f] not in s:
                        s[depth_copy[e][f]] = 1
                    else:
                        s[depth_copy[e][f]] += 1
            print(s)
            max_key = 0
            accuracy = max(s.values()) / sum(s.values())
            print(max(s.values()))
            print("the accuracy is %.4f" % accuracy)
            mleft.clear()
            mright.clear()
        cv2.imshow("result", image_copy)
        if k == 27:
            break
    cv2.imwrite("/home/pcms/Desktop/test1.png",image_copy)
    cv2.destroyAllWindows()
    rospy.loginfo("BW_getdepth end!")
