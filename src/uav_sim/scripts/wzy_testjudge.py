#!/usr/bin/python
#-*- encoding: utf8 -*-

# 模拟比赛过程，对裁判机的一个简单的验证程序
# 结合有限状态机的思想，将无人机的飞行过程分为多个状态：等待、导航、识别货物、降落、降落完成；
#其转移过程为：等待->导航->识别货物->导航—>导航->导航->识别货物->降落->降落完成
#当然也可以采取其他状态划分比如：穿第一个环，穿第二个环，穿第三个环，穿第四个环，穿第五个环，识别货物，降落等；
# 运行roslaunch uav_sim env.launch后，再在另一个终端中运行rosrun uav_sim judge.py即可开启裁判机；再运行rosrun uav_sim test_judeg.py即可开启验证程序
#该程序仅为示意裁判机功能，并非完成无人机比赛的任务，仅供参考。


from scipy.spatial.transform import Rotation as R
from collections import deque
from enum import Enum
import rospy
import cv2
import numpy as np
import math
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import apriltag

class TestNode:
    class FlightState(Enum):  # 飞行状态
        WAITING = 1
        NAVIGATING = 2
        DETECTING_TARGET = 3
        LANDING = 4
        LANDED = 5

    def __init__(self):
        rospy.init_node('Test_node', anonymous=True)
        rospy.logwarn('Test node set up.')
        
        # 无人机在世界坐标系下的位姿
        self.R_wu_ = R.from_quat([0, 0, 0, 1])
        self.t_wu_ = np.zeros([3], dtype=np.float64)

        self.image_ = None#前视相机图像
        self.imageSub_down = None#下视相机图像
        self.bridge_ = CvBridge()#图像转换

        self.flight_state_ = self.FlightState.WAITING#初始飞行状态为“等待”
        self.navigating_queue_ = deque()  # 存放多段导航信息的队列，队列元素为二元list，list的第一个元素代表导航维度（'x' or 'y' or 'z'），第二个元素代表导航目的地在该维度的坐标
        self.navigating_dimension_ = None  # 'x' or 'y' or 'z'
        self.navigating_destination_ = None
        self.navigating_xyr = None
        self.navigating_xy = None
        self.next_state_ = None  # 完成多段导航后将切换的飞行状态
        self.fly_state = None

        self.is_begin_ = False#由裁判机发布的开始信号
        self.ring_num_ = 0#穿过的圆环编号
        self.target_result_ = None#识别到的货物
        self.camera_info = None #相机内参

        self.publishCommand_ = rospy.Publisher('/m3e/cmd_string', String, queue_size=100)  # 发布无人机控制信号
        self.poseSub_ = rospy.Subscriber('/m3e/states', PoseStamped, self.poseCallback)  # 接收处理无人机位姿信息，只允许使用姿态信息
        self.imageSub_ = rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.imageCallback)  # 接收下视摄像头图像
        self.imageSub_down = rospy.Subscriber('/iris/usb_cam_down/image_raw', Image, self.imageCallback_down)  # 接收下视摄像头图像       
        self.BoolSub_ = rospy.Subscriber('/m3e/cmd_start', Bool, self.startcommandCallback)  # 接收开始飞行的命令
        self.ringPub_ = rospy.Publisher('/m3e/ring', String, queue_size=100)#发布穿过圆环信号
        self.targetPub_ = rospy.Publisher('/m3e/target_result', String, queue_size=100)#发布识别到的货物信号
        self.cameraSub = rospy.Subscriber('/iris/usb_cam/camera_info', CameraInfo, self.camera_info_callback) # 相机内参信号

        rate = rospy.Rate(0.3)#控制频率
        while not rospy.is_shutdown():
            if self.is_begin_:#发布开始信号后，开始进行决策
                self.decision()
            if self.flight_state_ == self.FlightState.LANDED:
                break
            rate.sleep()
        rospy.logwarn('Test node shut down.')

    # 按照一定频率，根据无人机的不同状态进行决策，并发布无人机控制信号
    def decision(self):
        if self.flight_state_ == self.FlightState.WAITING:  # 等待裁判机发布开始信号后，起飞
            rospy.logwarn('State: WAITING')
            self.publishCommand('takeoff')
            self.navigating_queue_ = deque([['y', 1.8]])#将无人机下次移动的目标设为y=1.8
            self.switchNavigatingState()#调用状态转移函数
            self.flight_state_=self.FlightState.NAVIGATING#下一个状态为“导航”



        elif self.flight_state_ == self.FlightState.NAVIGATING:#无人机根据视觉定位导航飞行
            rospy.logwarn('State: NAVIGATING')
            #根据导航信息发布无人机控制命令
            self.switchNavigatingState()#调用状态转移函数


        elif self.flight_state_ == self.FlightState.DETECTING_TARGET:#无人机来到货架前，识别货物
            rospy.logwarn('State: DETECTING_TARGET')
            if self.detectTarget():#如果检测到了货物，发布识别到的货物信号
                rospy.loginfo('Target detected.')
                if len(self.target_result_) == 2:#检测完所有货物后，发布识别到的货物信号
                    self.targetPub_.publish(self.target_result_)
                self.flight_state_=self.FlightState.NAVIGATING#下一个状态为“导航”
            else:#若没有检测到货物，则采取一定的策略，继续寻找货物
                self.switchNavigatingState()
                pass


        elif self.flight_state_ == self.FlightState.LANDING:#无人机穿过第五个圆环，开始降落
            rospy.logwarn('State: LANDING')
            #根据导航信息发布无人机控制命令
            #...
            #假如此时已经调整到指定位置，则降落
            self.publishCommand('land')
            self.flight_state_=self.FlightState.LANDED#此时无人机已经成功降落
            
        else:
            pass

    # 在飞行过程中，更新导航状态和信息
    def switchNavigatingState(self):
             # 从队列头部取出无人机下一次导航的状态信息
            # next_nav = self.navigating_queue_.popleft()
            if self.flight_state_ == self.FlightState.NAVIGATING:#如果当前状态为“导航”，则处理self.image_，得到无人机当前位置与圆环的相对位置，更新下一次导航信息和飞行状态
                self.fly_state = 'ring'
                if self.ring_num_ == 2:
                    self.fly_w()
                self.fly_p()
            #假如此时已经穿过了圆环，则发出相应的信号
                self.ring_num_ = self.ring_num_ + 1
            #判断是否已经穿过圆环
                if self.ring_num_ > 0:
                    self.ringPub_.publish('ring '+str(self.ring_num_))
                    self.next_state_=self.FlightState.NAVIGATING
            #如果穿过了第一个或第四个圆环，则下一个状态为“识别货物”
                    if self.ring_num_ == 1 or self.ring_num_ ==4 :
                        self.next_state_ = self.FlightState.DETECTING_TARGET
                    if self.ring_num_== 5:
                        self.next_state_ = self.FlightState.LANDING
            if self.flight_state_ == self.FlightState.DETECTING_TARGET:#如果当前状态为“识别货物”，则采取一定策略进行移动，更新下一次导航信息和飞行状态
                #...
                pass

            if self.flight_state_ == self.FlightState.LANDING:#如果当前状态为“降落”，则处理self.image_down，得到无人机当前位置与apriltag码的相对位置，更新下一次导航信息和飞行状态
                self.fly_state = 'apriltag'
                self.fly_p()
            self.flight_state_=self.next_state_#更新飞行状态

    # 判断是否检测到目标
    def detectTarget(self):
        # if self.image_ is None:
        #     return False
        # image_copy = self.image_.copy()
        #处理前视相机图像，检测货物
        #...
        #若检测到了货物，则发布识别到的货物信号,例如识别到了红色和黄色小球，则发布“ry”
        if self.ring_num_==1:
            self.target_result_='y'
        if self.ring_num_==4:
          self.target_result_='yr'
        return True
        #若没有检测到货物，则返回False
        return False

    #飞行函数
    def fly_p(self):
        while not rospy.is_shutdown():
            #调整无人机姿态：如果yaw与90度相差超过正负10度，需要进行旋转调整yaw
            (yaw, pitch, roll) = self.R_wu_.as_euler('zyx', degrees=True)
            if self.ring_num_ < 3: 
                yaw_diff = yaw - 90 if yaw > -90 else yaw + 270
            elif self.ring_num_ == 2:
                yaw_diff = yaw + 180 if yaw < 0 else yaw - 180
            else:
                yaw_diff = yaw + 90 if yaw < 90 else yaw - 270
            rospy.loginfo("yawdiff=%d",int(yaw_diff))
            if yaw_diff > 10:  # clockwise
                self.publishCommand('cw %d' % (int(yaw_diff) if yaw_diff > 15 else 15))
                rospy.sleep(5)
            elif yaw_diff < -10:  # counterclockwise
                self.publishCommand('ccw %d' % (int(-yaw_diff) if yaw_diff < -15 else 15))
                rospy.sleep(5)
           #进行x, y方向的平移调整
            navigation_cp=self.navigating_xyr.astype(int) if self.fly_state == 'ring' else self.navigating_xy.astype(int)
            if navigation_cp[2]<120 and self.fly_state == 'ring':
                if navigation_cp[0]>=10:
                    self.publishCommand('right '+str(navigation_cp[0]))
                    rospy.sleep(5)
                if navigation_cp[0]<=-10:
                    self.publishCommand('left '+str(-navigation_cp[0]))
                    rospy.sleep(5)
                if navigation_cp[1]>=10:
                    self.publishCommand('down '+str(navigation_cp[1]))
                    rospy.sleep(5)
                if navigation_cp[1]<=-10:
                    self.publishCommand('up '+str(-navigation_cp[1]))
                    rospy.sleep(5)
            elif self.fly_state == 'apriltag':
                if navigation_cp[0]>=10:
                    self.publishCommand('right '+str(navigation_cp[0]))
                    rospy.sleep(5)
                if navigation_cp[0]<=-10:
                    self.publishCommand('left '+str(-navigation_cp[0]))
                    rospy.sleep(5)
                if navigation_cp[1]>=10 and self.flight_state_ == self.FlightState.LANDING:
                    self.publishCommand('back '+str(navigation_cp[1]))
                    rospy.sleep(5)
                if navigation_cp[1]<=-10 and self.flight_state_ == self.FlightState.LANDING:
                    self.publishCommand('forward '+str(-navigation_cp[1]))
                    rospy.sleep(5)
            #前进一段距离
            # rospy.sleep(5)
            if self.fly_state == 'ring':
                if navigation_cp[2]<100:
                    self.publishCommand('forward 100')
                    rospy.sleep(5)
                #穿过圆环
                if navigation_cp[2]>100:
                    self.publishCommand("forward 200")
                    rospy.sleep(5)
                    # self.publishCommand("land")
                    break
            elif self.flight_state_ == self.FlightState.LANDING:
                #离地面足够近时，降落
                if LAND==1:
                    rospy.sleep(3)
                    self.publishCommand("land")
                    break
                #下降一段距离
                else:
                    self.publishCommand('down 80')
                    rospy.sleep(5)
                LAND=LAND-1
            else:
                self.publishCommand('forward 100')
                rospy.sleep(5) 
                break

    def fly_w(self):
        for i in range(4):
            #调整无人机姿态：如果yaw与90度相差超过正负10度，需要进行旋转调整yaw
            (yaw, pitch, roll) = self.R_wu_.as_euler('zyx', degrees=True)
            if self.ring_num_ == 2 and i<2: # 90
                yaw_diff = yaw - 90 if yaw > -90 else yaw + 270
            elif (self.ring_num_ == 2 and i >= 2) or (self.ring_num_ == 3 and i<2): # 180
                yaw_diff = yaw + 180 if yaw < 0 else yaw - 180
            else: # -90
                yaw_diff = yaw + 90 if yaw < 90 else yaw - 270
            rospy.loginfo("yawdiff=%d",int(yaw_diff))
            if yaw_diff > 10:  # clockwise
                self.publishCommand('cw %d' % (int(yaw_diff) if yaw_diff > 15 else 15))
                rospy.sleep(5)
            elif yaw_diff < -10:  # counterclockwise
                self.publishCommand('ccw %d' % (int(-yaw_diff) if yaw_diff < -15 else 15))
                rospy.sleep(5)
           #进行x, y方向的平移调整
             #进行x，y方向的平移调整
            navigation_cp=self.navigating_xy.astype(int)
            if navigation_cp[0]>=10:
                self.publishCommand('right '+str(navigation_cp[0]))
                rospy.sleep(5)
            if navigation_cp[0]<=-10:
                self.publishCommand('left '+str(-navigation_cp[0]))
                rospy.sleep(5)
            if navigation_cp[1]>=0:
                self.publishCommand('back '+str(navigation_cp[1]))
                rospy.sleep(5)
            if navigation_cp[1]<=-30:
                self.publishCommand('forward '+str(-navigation_cp[1]))
                rospy.sleep(5)
            if i == 2:
                self.publishCommand('ccw 90')

    def imageCallback(self, msg):
        try:
            cv_img=self.bridge.imgmsg_to_cv2(msg,'bgr8')#将图片转换为opencv格式
            cv_img_cp=cv_img.copy()
            img_gray = cv2.cvtColor(cv_img_cp, cv2.COLOR_BGR2GRAY)#转换为灰度图
            img_g=cv2.GaussianBlur(img_gray, (3, 3), 0)#滤波
            circles = cv2.HoughCircles(img_g, cv2.HOUGH_GRADIENT, 1, 100, param1=80, param2=40, minRadius=20, maxRadius=150)  #霍夫圆检测
            #画圈
            if circles is not None:
                for i in circles[0,:]:
                    cv2.circle(cv_img_cp, (i[0], i[1]), i[2], (255, 0, 0), 3)  # 画圆
                    cv2.circle(cv_img_cp, (i[0], i[1]), 2, (255, 0, 0), 3)  # 画圆心
                    #输出圆心的图像坐标和半径
                    rospy.loginfo("( %d  ,  %d ),r=  %d ",i[0],i[1],i[2])
                    #更新导航信息，此处是粗略估计图像和实际距离，其实可以用结合目标在图像中的位置和相机内外参数得到较准确的坐标
                    self.navigating_xyr = np.array([(i[0] - self.camera_info[0]) / i[2] * 70, (i[1] - self.camera_info[1]) / i[2] * 70, i[2]])
    
        except CvBridgeError as e:
            print(e)
    # 二维码
    def imageCallback_down(self, msg):
        try:
            cv_img= self.bridge.imgmsg_to_cv2(msg,'bgr8')#将图片转换为opencv格式
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
                    x=(tag.center[0].astype(int)-self.camera_info[0])*75/length
                    y=(tag.center[1].astype(int)-self.camera_info[1])*75/length
                    self.navigating_xy=np.array([x,y])
        except CvBridgeError as e:
            print(e)  

    # 向相关topic发布无人机控制命令
    def publishCommand(self, command_str):
        msg = String()
        msg.data = command_str
        self.self.publishCommand_.publish(msg)
    # 接收无人机位姿 本课程只允许使用姿态信息
    def poseCallback(self, msg):
        self.t_wu_ = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.R_wu_ = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        pass
    # 接收开始信号
    def startcommandCallback(self, msg):
        self.is_begin_ = msg.data
    # 接收相机内参
    def camera_info_callback(self, msg):
        self.camera_info = [msg.K[2], msg.K[5]]

if __name__ == '__main__':
    cn = TestNode()


