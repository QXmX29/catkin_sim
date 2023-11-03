#!/usr/bin/python
#-*- encoding: utf8 -*-

#一个简单的穿环demo：利用hough圆检测识别图像中的圆，然后调整无人机姿态和位置，穿过圆心，降落；
#无人机移动逻辑是每前进1m，就调整一次姿态，然后再前进1m，当检测到圆的半径大于阈值时，认为到达环近点，向前飞行2m穿过环；
#运行roslaunch uav_sim demo2.launch后，再在另一个终端中运行rosrun uav_sim demo2.py 

import cv2
import rospy
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Bool
from collections import deque
import numpy as np



navigation=np.array([0,0,0])#x,y,r 导航目地信息
cv_img=None
R_wu_ = R.from_quat([0, 0, 0, 1])#无人机位姿

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
        image_target=cv_img.copy()#cv_img_cp
        
        image_target_gray = cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY)#转换为灰度图
        image_target_g=cv2.GaussianBlur(image_target_gray, (3, 3), 0)#滤波
        circles = cv2.HoughCircles(image_target_g, cv2.HOUGH_GRADIENT, 1, 100, param1=80, param2=40, minRadius=20, maxRadius=150)  #霍夫圆检测
        #画圈
        target = ''
        color_dict = {"r":[{"lower":np.array([0,43,46]), "upper":np.array([10,255,255])},\
                           {"lower":np.array([156,43,46]), "upper":np.array([180,255,255])}],
                      "y":[{"lower":np.array([26,43,46]), "upper":np.array([34,255,255])}],
                      "b":[{"lower":np.array([100,43,46]), "upper":np.array([124,255,255])}]}
        if circles is not None:
            image_hsv = cv2.cvtColor(image_target, cv2.COLOR_BGR2HSV)
            for i in circles[0,:]:
                cv2.circle(image_target, (i[0], i[1]), i[2], (125, 125, 125), 3)  # 画圆
                cv2.circle(image_target, (i[0], i[1]), 2, (125, 125, 125), 3)  # 画圆心
                #输出圆心的图像坐标和半径
                rospy.loginfo("( %d  ,  %d ),r=  %d ",i[0],i[1],i[2])
                for color in color_dict.keys():
                    for color_range in color_dict[color]:
                        mask = cv2.inRange(image_hsv, color_range["lower"], color_range["upper"])
                        # res = cv2.bitwise_and(image_target, image_target, mask)
                        srate = sum(mask/255)/(math.pi*i[2]**2)
                        rospy.loginfo(color+": "+str(srate))
                        if srate>1:
                            target += color
                navigation = np.array([(i[0] - 160) / i[2] * 70, (i[1] - 80) / i[2] * 70, i[2]])
            image_result_pub.publish(bridge.cv2_to_imgmsg(image_target,encoding='bgr8'))#cv_img_cp
    except CvBridgeError as e:
        print(e)

#飞行函数
def Fly():
    global navigation
    global R_wu_
    Commandpub('takeoff')#起飞
    rospy.sleep(5)
    while not rospy.is_shutdown():
        #调整无人机姿态：如果yaw与theta度相差超过正负10度，需要进行旋转调整yaw
        (yaw, pitch, roll) = R_wu_.as_euler('zyx', degrees=True)
        theta = 0
        yaw_diff = yaw - theta if yaw > theta-180 else yaw + 360 - theta
        rospy.loginfo("yawdiff=%d",int(yaw_diff))
        if yaw_diff > 10:  # clockwise
            Commandpub('cw %d' % (int(yaw_diff) if yaw_diff > 15 else 15))
            rospy.sleep(5)
        elif yaw_diff < -10:  # counterclockwise
            Commandpub('ccw %d' % (int(-yaw_diff) if yaw_diff < -15 else 15))
            rospy.sleep(5)
        #进行x，y方向的平移调整
        navigation_cp=navigation.astype(int)
        if navigation_cp[2]<120:
            if navigation_cp[0]>=10:
                Commandpub('right '+str(navigation_cp[0]))
                rospy.sleep(5)
            if navigation_cp[0]<=-10:
                Commandpub('left '+str(-navigation_cp[0]))
                rospy.sleep(5)
            if navigation_cp[1]>=10:
                Commandpub('down '+str(navigation_cp[1]))
                rospy.sleep(5)
            if navigation_cp[1]<=-10:
                Commandpub('up '+str(-navigation_cp[1]))
                rospy.sleep(5)
        #前进一段距离
        # rospy.sleep(5)
        if navigation_cp[2]<100:
            Commandpub('forward 100')
            rospy.sleep(5)
        #判断是否到达终点
        if navigation_cp[2]>100:
            Commandpub("forward 200")
            rospy.sleep(5)
            Commandpub("land")
            break

#发布控制信号
def Commandpub(str):
    msg=String()
    msg.data=str    
    command_pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('demo2')
    rospy.loginfo('demo2 node set up')

    image_sub=rospy.Subscriber('/iris/usb_cam/image_raw',Image,imagecallback)
    poseSub_ = rospy.Subscriber('/m3e/states', PoseStamped, poseCallback)
    image_result_pub=rospy.Publisher('/get_images/image_result_circle',Image,queue_size=10)
    command_pub=rospy.Publisher('/m3e/cmd_string',String,queue_size=100)
    rospy.sleep(2)
    Fly()
