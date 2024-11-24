#!/usr/bin/env python3
import rospy
from mr_voice.msg import Voice
from std_msgs.msg import String
import os
def callback_voice(msg):
    global s
    s = msg.text
def say(text):
    os.system(f'espeak "{text}"')
    rospy.loginfo(text)
if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    '''
    s = ""
    print("speaker")
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    while not rospy.is_shutdown():
        print("spek",s)
    '''
    say("hello") 
    say("")

