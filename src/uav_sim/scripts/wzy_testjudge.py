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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


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
        self.next_state_ = None  # 完成多段导航后将切换的飞行状态

        self.is_begin_ = False#由裁判机发布的开始信号
        self.ring_num_ = 0#穿过的圆环编号
        self.target_result_ = None#识别到的货物

        self.commandPub_ = rospy.Publisher('/m3e/cmd_string', String, queue_size=100)  # 发布无人机控制信号
        self.poseSub_ = rospy.Subscriber('/m3e/states', PoseStamped, self.poseCallback)  # 接收处理无人机位姿信息，只允许使用姿态信息
        self.imageSub_ = rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.imageCallback)  # 接收下视摄像头图像
        self.imageSub_down = rospy.Subscriber('/iris/usb_cam_down/image_raw', Image, self.imageCallback_down)  # 接收下视摄像头图像       
        self.BoolSub_ = rospy.Subscriber('/m3e/cmd_start', Bool, self.startcommandCallback)  # 接收开始飞行的命令
        self.ringPub_ = rospy.Publisher('/m3e/ring', String, queue_size=100)#发布穿过圆环信号
        self.targetPub_ = rospy.Publisher('/m3e/target_result', String, queue_size=100)#发布识别到的货物信号

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
            #...    
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
            #...
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
                #...
                pass
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



    # 向相关topic发布无人机控制命令
    def publishCommand(self, command_str):
        msg = String()
        msg.data = command_str
        self.commandPub_.publish(msg)
    # 接收无人机位姿 本课程只允许使用姿态信息
    def poseCallback(self, msg):
        self.t_wu_ = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.R_wu_ = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        pass
    # 接收前视相机图像
    def imageCallback(self, msg):
        try:
            self.image_ = self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as err:
            print(err)
    #接受下视相机图像
    def imageCallback_down(self,msg):
        try:
            self.image_down = self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as err:
            print(err)
    # 接收开始信号
    def startcommandCallback(self, msg):
        self.is_begin_ = msg.data


if __name__ == '__main__':
    cn = TestNode()


