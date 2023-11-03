#!/usr/bin/python
#-*- encoding: utf8 -*-

#一个简单的识别二维码降落sdemo：利用apriltag.Detector检测识别图像中的二维码，然后调整无人机姿态和位置，降落；
#无人机移动逻辑是前进到二维码附近的位置，检测到二维码就调整一次姿态，然后再下降0.8m，直到降落；
#运行roslaunch uav_sim demo1.launch后，再在另一个终端中运行rosrun uav_sim demo1.py 

import cv2
import rospy
import apriltag
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Bool
from collections import deque
import numpy as np



navigation=np.array([0,0])#x,y导航目地信息
cv_img=None
R_wu_ = R.from_quat([0, 0, 0, 1])#无人机位姿
LAND=2

bridge=CvBridge()

#获取无人机位姿回调函数
def poseCallback(msg):
        global R_wu_
        R_wu_ = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        pass

#获取图像回调函数：霍夫检测圆并更新navigation导航信息
def imagecallback(data):
    global cv_img
    global navigation
    try:
        cv_img=bridge.imgmsg_to_cv2(data,'bgr8')#将图片转换为opencv格式
        cv_img_cp=cv_img.copy()
        img_gray = cv2.cvtColor(cv_img_cp, cv2.COLOR_BGR2GRAY)#转换为灰度图
        img_g=cv2.GaussianBlur(img_gray, (3, 3), 0)#滤波
        at_detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11 tag25h9'))
        tags = at_detector.detect(img_g)
        if tags is not None:
          for tag in tags:
            # for i in range(4):
            #     cv2.circle(cv_img_cp, tuple(tag.corners[i].astype(int)), 4, (255, 0, 0), 2)
            cv2.circle(cv_img_cp, tuple(tag.center.astype(int)), 4, (2, 180, 200), 4)
            length=tag.corners[1].astype(int)[0]-tag.corners[0].astype(int)[0]
            x=(tag.center[0].astype(int)-160)*60/length
            y=(tag.center[1].astype(int)-120)*60/length
            navigation=np.array([x,y])

          image_result_pub.publish(bridge.cv2_to_imgmsg(cv_img_cp,encoding='bgr8'))  

    except CvBridgeError as e:
        print(e)


#飞行函数
def Fly():
    global navigation
    global R_wu_
    global LAND
    Commandpub('takeoff')#起飞
    rospy.sleep(10)
    Commandpub('forward 200')
    rospy.sleep(5)
    while not rospy.is_shutdown():
            #调整无人机姿态：如果yaw与90度相差超过正负10度，需要进行旋转调整yaw
            (yaw, pitch, roll) = R_wu_.as_euler('zyx', degrees=True)
            yaw_diff = yaw - 90 if yaw > -90 else yaw + 270
            rospy.loginfo("yawdiff=%d",int(yaw_diff))
            if yaw_diff > 10:  # clockwise
                Commandpub('cw %d' % (int(yaw_diff) if yaw_diff > 15 else 15))
                rospy.sleep(5)
            elif yaw_diff < -10:  # counterclockwise
                Commandpub('ccw %d' % (int(-yaw_diff) if yaw_diff < -15 else 15))
                rospy.sleep(5)

           #进行x，y方向的平移调整
            navigation_cp=navigation.astype(int)
            if navigation_cp[0]>=10:
                Commandpub('right '+str(navigation_cp[0]))
                rospy.sleep(5)
            if navigation_cp[0]<=-10:
                Commandpub('left '+str(-navigation_cp[0]))
                rospy.sleep(5)
            if navigation_cp[1]>=10:
                Commandpub('back '+str(navigation_cp[1]))
                rospy.sleep(5)
            if navigation_cp[1]<=-10:
                Commandpub('forward '+str(-navigation_cp[1]))
                rospy.sleep(5)
            #离地面足够近时，降落
            if LAND==1:
                rospy.sleep(3)
                Commandpub("land")
                break
            #下降一段距离
            else:
                Commandpub('down 80')
                rospy.sleep(5)
            LAND=LAND-1


#发布控制信号
def Commandpub(str):
    msg=String()
    msg.data=str    
    command_pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('demo1')
    rospy.loginfo('demo1 node set up')

    image_sub=rospy.Subscriber('/iris/usb_cam_down/image_raw',Image,imagecallback)#订阅下视相机图像
    poseSub_ = rospy.Subscriber('/m3e/states', PoseStamped, poseCallback)
    image_result_pub=rospy.Publisher('/get_images/image_result_code',Image,queue_size=10)#发布图像结果
    command_pub=rospy.Publisher('/m3e/cmd_string',String,queue_size=100)
    rospy.sleep(2)
    Fly()
