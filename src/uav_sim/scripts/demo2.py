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
import numpy as np
import math

from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo

from cv_bridge import CvBridge, CvBridgeError
import cv2
import apriltag
import random


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
        self.image_down = None#下视相机图像
        self.bridge_ = CvBridge()#图像转换
        # 临时flag，只是想查看传输图像的长宽/相机参数
        self.temp_flag = True
        self.temp_flag_down = True
        self.temp_flag_cam = True
        # 尝试得到相机参数
        self.caminfo_ = None
        # 临时flag，每次检测货物，机身仅下降一次
        self.detect_state = 0
        # 飞行控制参数
        self.min_dis_dict = {'x': 20, 'y': 20, 'z': 20, 'r': 1}
        self.max_dis_dict = {'x': 500, 'y': 500, 'z': 500, 'r': 360}
        self.nav_dict = {'x': ["back ", "forward "],
                    'y': ["left ", "right "],
                    'z': ["down ", "up "],
                    'r': ["ccw", "cw"]}#旋转，逆时针为-顺时针为+，采用degree单位

        self.flight_state_ = self.FlightState.WAITING#初始飞行状态为“等待”
        self.navigating_queue_ = None#deque()  # 存放多段导航信息的队列，队列元素为二元list，list的第一个元素代表导航维度（'x' or 'y' or 'z'），第二个元素代表导航目的地在该维度的坐标
        self.navigating_dimension_ = None  # 'x' or 'y' or 'z'
        self.navigating_destination_ = None
        self.next_state_ = None  # 完成多段导航后将切换的飞行状态

        self.is_begin_ = True#由裁判机发布的开始信号
        self.ring_num_ = 0#穿过的圆环编号
        self.target_result_ = None#识别到的货物

        self.commandPub_ = rospy.Publisher('/m3e/cmd_string', String, queue_size=100)  # 发布无人机控制信号
        self.poseSub_ = rospy.Subscriber('/m3e/states', PoseStamped, self.poseCallback)  # 接收处理无人机位姿信息，只允许使用姿态信息
        self.imageSub_ = rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.imageCallback)  # 接收前视摄像头图像
        self.imageSub_down = rospy.Subscriber('/iris/usb_cam_down/image_raw', Image, self.imageCallback_down)  # 接收下视摄像头图像
        self.BoolSub_ = rospy.Subscriber('/m3e/cmd_start', Bool, self.startcommandCallback)  # 接收开始飞行的命令
        self.ringPub_ = rospy.Publisher('/m3e/ring', String, queue_size=100)#发布穿过圆环信号
        self.targetPub_ = rospy.Publisher('/m3e/target_result', String, queue_size=100)#发布识别到的货物信号
        # 尝试获取相机参数
        self.caminfoSub_ = rospy.Subscriber('/iris/usb_cam/camera_info', CameraInfo, self.caminfoCallback)
        
        self.image_tag_pub = rospy.Publisher('/get_images/image_result_code',Image,queue_size=10)#发布图像结果
        self.image_circle_pub = rospy.Publisher('/get_images/image_result_circle',Image,queue_size=10)

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
            rospy.sleep(1.5)
            self.publishCommand('takeoff')
            rospy.loginfo("Take off!")
            rospy.sleep(4)
            self.navigating_queue_ = deque([['z', 100]])
            self.navigating_queue_.append(['y', -120])
            self.navigating_queue_.append(['x', 700])
            self.navigating_queue_.append(['r', -90])
            self.navigating_queue_.append(['x', 130])
            self.navigating_queue_.append(['r', 180])
            self.navigating_queue_.append(['y', 190])
            self.navigating_queue_.append(['r', 90])
            self.switchNavigatingState()#调用状态转移函数
            # self.flight_state_=self.FlightState.NAVIGATING#下一个状态为“导航”


        elif self.flight_state_ == self.FlightState.NAVIGATING:#无人机根据视觉定位导航飞行
            rospy.logwarn('State: NAVIGATING')
            while len(self.navigating_queue_)>0:
                next_nav = self.navigating_queue_.popleft()# 从队列头部取出无人机下一次导航的状态信息
                self.Fly(next_nav)
            self.switchNavigatingState()#调用状态转移函数


        elif self.flight_state_ == self.FlightState.DETECTING_TARGET:#无人机来到货架前，识别货物
            rospy.logwarn('State: DETECTING_TARGET')
            if self.detectTarget():#如果检测到了货物，发布识别到的货物信号
                rospy.loginfo('Target detected.')
                self.navigating_queue_.append(['z', 110])
                self.detect_state = 0
                if len(self.target_result_) == 2:#检测完所有货物后，发布识别到的货物信号
                    self.targetPub_.publish(self.target_result_)
                self.flight_state_=self.FlightState.NAVIGATING#下一个状态为“导航”
            else:#若没有检测到货物，则采取一定的策略，继续寻找货物
                self.switchNavigatingState()


        elif self.flight_state_ == self.FlightState.LANDING:#无人机穿过第五个圆环，开始降落
            rospy.logwarn('State: LANDING')
            #根据导航信息发布无人机控制命令
            #...
            #假如此时已经调整到指定位置，则降落
            self.publishCommand('land')
            self.flight_state_=self.FlightState.LANDED#此时无人机已经成功降落
            
        else:
            pass

    def Fly(self, next_nav):
        # 根据导航信息发布无人机控制命令
        magnitude = next_nav[1]
        if magnitude==0:
            self.publishCommand("stop")
            return
        direction = self.nav_dict[next_nav[0]][(magnitude>0)*1]
        dmove = self.max_dis_dict[next_nav[0]]
        # 每次只移动dx/dy/dz cm，多余的存回/重新计算，移动后即检查euler是否稳定(符合预期)
        magnitude = int(round(math.fabs(magnitude)))
        while magnitude>dmove:
            t_begin = rospy.Time.now().to_sec()
            if next_nav[0]!='r':
                # 规定走直角路线，每次都要规范yaw
                self.Adjust()
            rospy.loginfo("dt_adjust="+str(rospy.Time.now().to_sec()-t_begin))
            t_begin = rospy.Time.now().to_sec()
            self.publishCommand(direction+str(dmove))
            magnitude -= dmove
            rospy.sleep(max(1.5, float(dmove/10)))
            rospy.loginfo("dt_dmove="+str(rospy.Time.now().to_sec()-t_begin))
        if magnitude<self.min_dis_dict[next_nav[0]]:
            self.publishCommand(self.nav_dict[next_nav[0]][(magnitude<0)*1]\
                                + str(self.min_dis_dict[next_nav[0]]))
            rospy.sleep(max(1.5, float(self.min_dis_dict[next_nav[0]]/10)))
            self.publishCommand(direction+str(int(self.min_dis_dict[next_nav[0]]+magnitude)))
            rospy.sleep(max(1.5, float(self.min_dis_dict[next_nav[0]]+magnitude)))
        else:
            self.publishCommand(direction+str(magnitude))
        rospy.sleep(max(1.5, float(magnitude/10)))
    
    def Adjust(self):
        adjusted = False
        while not adjusted:
            (yaw, pitch, roll) = self.R_wu_.as_euler('zyx', degrees=True)
            if math.fabs(yaw%90)>4:
                rospy.loginfo("Adjust yaw (from yaw="+str(yaw)+" to dest="+str(round(yaw/90)*90))
                d_yaw = -(round(yaw/90)*90 - yaw)
                adjust = self.nav_dict['r'][(d_yaw>0)*1]
                drot = self.max_dis_dict['r']
                d_yaw = int(round(math.fabs(d_yaw))+2)
                while d_yaw>drot:
                    self.publishCommand(adjust+str(drot))
                    rospy.sleep(max(1.5, float(drot/10)))
                    d_yaw -= drot
                if d_yaw<self.min_dis_dict['r']:
                    self.publishCommand(self.nav_dict['r'][(d_yaw<0)*1]\
                                        +str(self.min_dis_dict['r']))
                    rospy.sleep(max(1.5, float(drot/10)))
                    self.publishCommand(adjust+str(self.min_dis_dict['r']+d_yaw))
                    rospy.sleep(max(1.5, float((self.min_dis_dict['r']+d_yaw)/10)))
                else:
                    self.publishCommand(adjust+str(d_yaw))
                    rospy.sleep(max(1.5, int(round(d_yaw/10))))
            else:
                adjusted = True
        # 悬停的同时应该可以调整好pitch和roll
        self.publishCommand("stop")
        rospy.sleep(0.2)
    
    # 在飞行过程中，更新导航状态和信息
    def switchNavigatingState(self):
        if self.flight_state_ == self.FlightState.WAITING:
            self.next_state_ = self.FlightState.NAVIGATING
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
            self.Fly(['r', 0])#stop
            # 策略1.0: 利用下视摄像头瞄准二维码，上下左右移动配合机身旋转进行扫描
            try:
                # 检测二维码
                image_down_cp = self.image_down.copy()
                image_down_gray = cv2.cvtColor(image_down_cp, cv2.COLOR_BGR2GRAY)#转换为灰度图
                image_down_g = cv2.GaussianBlur(image_down_gray, (3, 3), 0)#滤波
                at_detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11 tag25h9'))
                tags = at_detector.detect(image_down_g)
                if tags is not None:
                    # 其实应该只能检测到一个
                    if len(tags)>1:
                        rospy.logwarn("TOO MANY TAGS! (DETECTED "+str(len(tags))+" TAGS)!")
                    for tag in tags:
                        cv2.circle(image_down_cp, tuple(tag.center.astype(int)), 4, (2, 180, 200), 4)
                        # Apriltag的边长length是相机图像中的像素差?
                        length=tag.corners[1].astype(int)[0]-tag.corners[0].astype(int)[0]
                        rospy.loginfo("Apriltag: length="+str(length))
                        # 相对图片中心的距离<->相对此时无人机左右偏移二维码的距离(==0)
                        th_LR = 0
                        th_FB = 0
                        x = -(tag.center[1].astype(int)-(120+th_FB))*60/length # x为前方, forward/back
                        y = -(tag.center[0].astype(int)-(160+th_LR))*60/length # y为左侧, left/right
                        # 必要时调整位置
                        if math.fabs(x)>0.1:
                            self.Fly(['x', x])
                        if math.fabs(y)>0.1:
                            self.Fly(['y', y])
                        # 根据图像估计离地面的距离
                    self.image_tag_pub.publish(self.bridge_.cv2_to_imgmsg(image_down_cp, encoding='bgr8'))
            except CvBridgeError as e:
                print(e)
            # 每轮均以二维码为中心左右移动100cm以内，每轮结束后上下移动约20cm
            # 每次移动均调整yaw和pitch进行扫描
            # (yaw, pitch, roll) = self.R_wu_.as_euler('zyx', degrees=True)
            # 调整高度
            if self.detect_state==0:#下降
                self.Fly(['z', -110])
            elif self.detect_state==1:#逆时针旋转
                self.Fly(['r', -60])
            elif self.detect_state==2:#顺时针旋转
                self.Fly(['r', 120])
            self.detect_state += 1
            # self.next_state_ = self.FlightState.DETECTING_TARGET

        if self.flight_state_ == self.FlightState.LANDING:#如果当前状态为“降落”，则处理self.image_down，得到无人机当前位置与apriltag码的相对位置，更新下一次导航信息和飞行状态
            #...
            pass
        self.flight_state_=self.next_state_#更新飞行状态

    # 判断是否检测到目标
    def detectTarget(self):
        self.Fly(['r', 0])#stop
        #版本2.0: 一次性判断，无坐标追踪&反复确认
        if self.image_ is None:
            return False
        image_target = self.image_.copy()
        #处理前视相机图像，检测货物
        image_target_rgb = cv2.cvtColor(image_target, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image_target_rgb, cv2.COLOR_RGB2HSV)
        # edges = cv2.Canny(image_target, 100, 200) # Canny边缘检测
        #由于货物边缘不如圆环清晰，所以先尝试锁定颜色
        target = ''
        color_dict = {"r":[{"lower":np.array([0,43,46]), "upper":np.array([10,255,255])},\
                           {"lower":np.array([156,43,46]), "upper":np.array([180,255,255])}],
                      "y":[{"lower":np.array([26,43,46]), "upper":np.array([34,255,255])}],
                      "b":[{"lower":np.array([100,43,46]), "upper":np.array([124,255,255])}]}
        for color in color_dict.keys():
            for color_range in color_dict[color]:
                mask = cv2.inRange(image_hsv, color_range["lower"], color_range["upper"])
                img = mask.copy()
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))
                erosion = cv2.erode(img, kernel, iterations=2)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50))
                dilation = cv2.dilate(erosion, kernel, iterations=1)
                mask = dilation
                mask_g = cv2.GaussianBlur(mask, (3, 3), 0)#滤波
                circles = cv2.HoughCircles(mask_g, cv2.HOUGH_GRADIENT, 1, 100, param1=80, param2=40, minRadius=20, maxRadius=150)  #霍夫圆检测
                if circles is not None:
                    circles = circles.astype(int)
                    for i in circles[0,:]:
                        cv2.circle(image_target, (i[0], i[1]), i[2], (125, 125, 125), 3)  # 画圆
                        cv2.circle(image_target, (i[0], i[1]), 2, (125, 125, 125), 3)  # 画圆心
                        #输出圆心的图像坐标和半径
                        rospy.loginfo("( %d , %d ), r= %d", i[0], i[1], i[2])
                        self.image_circle_pub.publish(self.bridge_.cv2_to_imgmsg(image_target,encoding='bgr8'))
                        #内接正方形像素点求和取平均，查看是否为圆环
                        len_a = round(i[2]/math.sqrt(2))-1
                        inscribed_square = mask[(i[0]-len_a):(i[0]+len_a), (i[1]-len_a):(i[1]+len_a)]
                        occupy_rate = sum(sum((inscribed_square.astype(int))/255))/inscribed_square.size
                        #设定阈值，占比大于阈值的是球而非圆环
                        if occupy_rate>0.7:
                            rospy.logwarn("Detected ball with color \'%s\'~", color)
                else:
                    #寻找轮廓和外接圆并绘制
                    binaries, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours)>0:
                        img_zero = np.zeros(np.shape(image_target), dtype=np.uint8)
                        image_target_res = cv2.add(image_target_rgb, img_zero, mask=mask)#添加掩码
                        dst = image_target_res.copy()
                        dst = cv2.drawContours(dst, contours, -1, (125, 125, 125), 3)
                        (cx, cy), cr = cv2.minEnclosingCircle(contours[0])
                        dst = cv2.circle(dst, (int(cx), int(cy)), int(cr), (201, 172, 221), 2)
                        #在空白图上画实心圆
                        enclosing_circle = img_zero.copy()
                        enclosing_circle = cv2.circle(enclosing_circle, (int(cx), int(cy)), int(cr), (255, 255, 255), -1)
                        img_base = enclosing_circle[:,:,0]#都是255所以随便一个维度都行
                        img_res = cv2.add(mask, img_zero[:,:,0], mask=img_base)#添加掩码
                        cbase = (sum(sum(img_base.astype(int)/255)))
                        ctarget = (sum(sum(img_res.astype(int)/255)))
                        occupy_rate = float(ctarget)/float(cbase)
                        rospy.loginfo("img.shape: "+str(img_res.shape)+", "+str(img_base.shape))
                        rospy.loginfo("%f/%f=%f ", ctarget, cbase, occupy_rate)
                        self.image_circle_pub.publish(self.bridge_.cv2_to_imgmsg(dst,encoding='bgr8'))
                        #设定阈值，占比大于阈值的很可能是货物
                        if occupy_rate>0.6:
                            rospy.loginfo("Detected ball with color \'%s\'~", color)
                        else:
                            pass
        #若检测到了货物，则发布识别到的货物信号,例如识别到了红色和黄色小球，则发布“ry”
        if len(target)==1:
            if self.ring_num_==1:
                self.target_result_ = target
            if self.ring_num_==4:
                self.target_result_ += target
            return True
        elif len(target)>0:
            rospy.loginfo("Too many circles? target="+target)
        #若没有检测到货物，则返回False
        else:
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
            if self.temp_flag:
                rospy.logwarn("CAM: H="+str(msg.height)+", W="+str(msg.width))
                self.temp_flag = False
            self.image_ = self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as err:
            print(err)
    #接受下视相机图像
    def imageCallback_down(self,msg):
        try:
            if self.temp_flag_down:
                rospy.logwarn("CAM_DOWN: H="+str(msg.height)+", W="+str(msg.width))
                self.temp_flag_down = False
            self.image_down = self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as err:
            print(err)
    # 接收开始信号
    def startcommandCallback(self, msg):
        self.is_begin_ = msg.data
    # 尝试接受相机参数
    def caminfoCallback(self, info):
        self.caminfo_ = {"D": info.D, "K": info.K, "R": info.R, "P": info.P,\
                         "H": info.height, "W": info.width, "ROI": info.roi}
        if self.temp_flag_cam:
            rospy.logwarn("CAM_INFO: "+str(self.caminfo_))
            rospy.loginfo(str(info))
            self.temp_flag_cam = False


if __name__ == '__main__':
    cn = TestNode()



