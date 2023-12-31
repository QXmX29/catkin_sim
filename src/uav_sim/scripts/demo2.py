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
        
        # 图像识别相关函数、信息等
        self.camera_info = None #相机内参
        self.image_ = None#前视相机图像
        self.imageSub_down = None#下视相机图像
        self.bridge_ = CvBridge()#图像转换
        
        # 飞机状态控制变量
        self.flight_state_ = self.FlightState.WAITING#初始飞行状态为“等待”
        self.next_state_ = None  # 完成多段导航后将切换的飞行状态

        # 导航相关：位置、角度变量等
        self.navigating_queue_ = deque()  # 存放多段导航信息的队列，队列元素为二元list，list的第一个元素代表导航维度（'x' or 'y' or 'z'），第二个元素代表导航目的地在该维度的坐标
        self.detect_state = 0   # 检测货物: 0机身下降; 1逆时针; 2顺时针
        self.theta = 0  # yaw角度控制: 0 --> 90 --> 180
        # 移动距离、货物颜色阈值参数
        self.min_dis_dict = {'x': 20, 'y': 20, 'z': 20, 'r': 10}
        self.max_dis_dict = {'x': 500, 'y': 500, 'z': 500, 'r': 360}
        self.nav_dict = {'x': ["back ", "forward "],
                         'y': ["left ", "right "],
                         'z': ["down ", "up "],
                         'r': ["ccw ", "cw "]} # 旋转，逆时针为-顺时针为+，采用degree单位
        self.color_dict = {"r":[{"lower":np.array([0,43,46]), "upper":np.array([10,255,255])},\
                                {"lower":np.array([156,43,46]), "upper":np.array([180,255,255])}],
                           "y":[{"lower":np.array([26,43,46]), "upper":np.array([34,255,255])}],
                           "b":[{"lower":np.array([100,43,46]), "upper":np.array([124,255,255])}]}

        # 发送给裁判机的信息
        self.is_begin_ = True#由裁判机发布的开始信号
        self.ring_num_ = 0#穿过的圆环编号
        self.target_result_ = None#识别到的货物
        
        # 发布的图片
        self.image_forward = None
        self.image_down = None
        
        # 订阅与发布
        self.commandPub_ = rospy.Publisher('/m3e/cmd_string', String, queue_size=100)  # 发布无人机控制信号
        self.poseSub_ = rospy.Subscriber('/m3e/states', PoseStamped, self.poseCallback)  # 接收处理无人机位姿信息，只允许使用姿态信息
        self.imageSub_ = rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.imageCallback)  # 接收前视摄像头图像
        self.imageSub_down = rospy.Subscriber('/iris/usb_cam_down/image_raw', Image, self.imageCallback_down)  # 接收下视摄像头图像       
        self.BoolSub_ = rospy.Subscriber('/m3e/cmd_start', Bool, self.startcommandCallback)  # 接收开始飞行的命令
        self.ringPub_ = rospy.Publisher('/m3e/ring', String, queue_size=100)#发布穿过圆环信号
        self.targetPub_ = rospy.Publisher('/m3e/target_result', String, queue_size=100)#发布识别到的货物信号
        self.cameraSub = rospy.Subscriber('/iris/usb_cam/camera_info', CameraInfo, self.camera_info_callback) # 相机内参信号
        
        self.image_tag_pub = rospy.Publisher('/get_images/image_result_code',Image,queue_size=10)#发布图像结果
        self.image_circle_pub = rospy.Publisher('/get_images/image_result_circle',Image,queue_size=10)

        # 相当于main函数，即在落地前循环执行
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
        # 处理上次 switch 中可能传来的导航信息: 穿环前进;检测货物前降落
        rospy.logwarn("[Decision] new round: navigate")
        while len(self.navigating_queue_)>0:
            next_nav = self.navigating_queue_.popleft()
            rospy.loginfo("next_nav: "+str(next_nav))
            self.Fly(next_nav)
        
        if self.flight_state_ == self.FlightState.WAITING:  # 等待裁判机发布开始信号后，起飞
            rospy.logwarn('State: WAITING')
            rospy.sleep(1.5)
            self.publishCommand('takeoff')
            rospy.loginfo("Take off!")
            rospy.sleep(2)
            # 起飞后可能需要根据仿真情况调整上下高度?
            # self.navigating_queue_.append(['z', 20])
            self.next_state_ = self.FlightState.NAVIGATING
        
        elif self.flight_state_ == self.FlightState.NAVIGATING:#无人机根据视觉定位导航飞行
            rospy.logwarn('State: NAVIGATING')
            # 调用状态转移函数
        
        elif self.flight_state_ == self.FlightState.DETECTING_TARGET:#无人机来到货架前，识别货物
            rospy.logwarn('State: DETECTING_TARGET')
            if self.detect_state==0:#先下降
                self.navigating_queue_.append(['z', -110])
            elif self.detectTarget():#如果检测到了货物，发布识别到的货物信号
                rospy.loginfo('Target detected.')
                # 检测到货物后无人机上升，并重置detect_state(=0)以便下次检测货物
                self.navigating_queue_.append(['z', 110])
                self.detect_state = 0
                # 检测完所有货物后，发布识别到的货物信号
                if len(self.target_result_) == 2:
                    self.targetPub_.publish(self.target_result_)
                # 已经确定下一个状态为“导航”，不需要再次进入状态转移函数
                self.flight_state_=self.FlightState.NAVIGATING
            #若没有检测到货物，则采取一定的策略，继续寻找货物
            else:
                pass
        
        elif self.flight_state_ == self.FlightState.LANDING:#无人机穿过第五个圆环，开始降落
            rospy.logwarn('State: LANDING')
            # 离地面足够近时，降落
            if LAND==1:
                rospy.sleep(3)
                self.publishCommand("land")
                self.flight_state_=self.FlightState.LANDED#此时无人机已经成功降落
            # 否则只尝试下降一段距离
            else:
                self.navigating_queue_.append(['z', -80])
            LAND -= 1
        
        else:
            pass
        
        self.switchNavigatingState()#调用状态转移函数
    
    # 在飞行过程中，更新导航状态和信息
    def switchNavigatingState(self):
        if self.flight_state_ == self.FlightState.WAITING:
            self.next_state_ = self.FlightState.NAVIGATING
        if self.flight_state_ == self.FlightState.NAVIGATING:#如果当前状态为“导航”，则处理self.image_，得到无人机当前位置与圆环的相对位置，更新下一次导航信息和飞行状态
            # 可能需要先检测二维码，以保证距离合适
            distance = self.detectApriltag()
            # 好吧，起飞阶段莫得二维码
            if distance<200:
                pass    # 考虑在距离过短时适当上升
            # 然后需要调整位姿
            self.Adjust()
            # 检测圆环位置: 借助圆半径估计位置, 阈值100或许可以根据相机参数和圆环参数等微调
            cr = self.detectCircle()
            if cr<0:    # 未检测到圆环或出现异常
                # 异常值：-1为找不到图像；-2为检测不到圆；-3未为疑似未起飞()；-4为没有合适的圆
                if cr==-3:
                    self.next_state_ = self.FlightState.WAITING
                # 检测不到圆，可能高度不够（或离得太近）
                elif cr==-2:
                    pass
                else:
                    pass
            # (调试)如果cr过小，是不是检测到远处的圆环了?
            elif cr<30:
                rospy.loginfo("[switch-Navigating] cr="+str(cr))
            elif cr<100:  # 试探性地前进; 可能需要修改以应对穿环中/穿环后检测到其他圆环的情况
                self.navigating_queue_.append(['x', 100])
            else:
                # 距离合适，尝试前进(200可能需要结合cr和相机参数等修改)
                self.navigating_queue_.append(['x', 200])
                # 假如此时已经穿过了圆环，则发出相应的信号
                self.ring_num_ = self.ring_num_ + 1
            # 判断是否已经穿过圆环
            if self.ring_num_ > 0:
                self.ringPub_.publish('ring '+str(self.ring_num_))
                # self.next_state_=self.FlightState.NAVIGATING
                # 如果闯过第二或第三个圆环，则需要转弯
                if self.ring_num_ == 2 or self.ring_num_ == 3 :
                    self.theta = 90*(self.ring_num_-1)
                # 如果穿过了第一个或第四个圆环，则下一个状态为“识别货物”
                if self.ring_num_ == 1 or self.ring_num_ == 4 :
                    self.next_state_ = self.FlightState.DETECTING_TARGET
                # 最后一个圆环即可降落
                if self.ring_num_== 5:
                    self.next_state_ = self.FlightState.LANDING
        if self.flight_state_ == self.FlightState.DETECTING_TARGET:#如果当前状态为“识别货物”，则采取一定策略进行移动，更新下一次导航信息和飞行状态
            # 策略1.0: 利用下视摄像头瞄准二维码，上下移动配合机身旋转进行扫描
            # 检测二维码并对准
            self.Fly(['r', 0])  # stop
            distance = self.detectApriltag() # 可能可以用到距离二维码的距离
            # 如果距离过大可能需要检查是否下降
            if distance > 100:
                self.detect_state = 1
                self.navigating_queue_.append(['z', int(round(80-distance))])
                rospy.logwarn("[DetectTarget] distance="+str(distance)+", is command lost?")
            # 调整 yaw 进行扫描
            else:
                if self.detect_state==1:#逆时针旋转
                    self.navigating_queue_.append(['r', -60])
                elif self.detect_state==2:#顺时针旋转
                    self.navigating_queue_.append(['r', 120])
                else:   # 0-down-x-1
                    pass
                self.detect_state += 1
            # self.next_state_ = self.FlightState.DETECTING_TARGET
        
        if self.flight_state_ == self.FlightState.LANDING:#如果当前状态为“降落”，则处理self.image_down，得到无人机当前位置与apriltag码的相对位置，更新下一次导航信息和飞行状态
            distance = self.detectApriltag()
        
        self.flight_state_=self.next_state_#更新飞行状态

    # 检测二维码并估计距离
    def detectApriltag(self):
        try:
            image_down_cp = self.image_down_.copy()
            image_down_gray = cv2.cvtColor(image_down_cp, cv2.COLOR_BGR2GRAY)#转换为灰度图
            image_down_g = cv2.GaussianBlur(image_down_gray, (3, 3), 0)#滤波
            at_detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11 tag25h9'))
            tags = at_detector.detect(image_down_g)
            if tags is not None:
                # 其实应该只能检测到一个?
                if len(tags)>1:
                    rospy.logwarn("TOO MANY TAGS! (DETECTED "+str(len(tags))+" TAGS)!")
                for tag in tags:
                    cv2.circle(image_down_cp, tuple(tag.center.astype(int)), 4, (2, 180, 200), 4)
                    # Apriltag的边长length是相机图像中的像素差?
                    length=tag.corners[1].astype(int)[0]-tag.corners[0].astype(int)[0]
                    # rospy.loginfo("Apriltag: length="+str(length))
                    # 根据相机参数修改
                    y = (tag.center[0].astype(int)-self.camera_info[0])/length*75   # y为左侧, left-/right+
                    x = -(tag.center[1].astype(int)-self.camera_info[1])/length*75  # x为前方, back-/forward+
                    # 相对图片中心的距离<->相对此时无人机左右偏移二维码的距离(==0)
                    # th_LR = 0
                    # th_FB = 0
                    # x = -(tag.center[1].astype(int)-(120+th_FB))*60/length
                    # y = -(tag.center[0].astype(int)-(160+th_LR))*60/length
                    # 必要时调整位置
                    if math.fabs(y)>0.1:
                        self.navigating_queue_.append(['y', y])
                    if math.fabs(x)>0.1:
                        self.navigating_queue_.append(['x', x])
                    # 根据图像估计离地面的距离
                    (yaw, pitch, roll) = self.R_wu_.as_euler('zyx', degrees=True)
                    if math.fabs(pitch)<2 and math.fabs(roll)<2:
                        # 认为误差足够小，所以尝试估算距离；可能需要用到相机焦距等参数
                        return -1
                self.image_down = image_down_cp
                # self.image_tag_pub.publish(self.bridge_.cv2_to_imgmsg(image_down_cp, encoding='bgr8'))
            return -1
        except CvBridgeError as e:
            print(e)
            return -1
    
    # 区分是货物还是圆环
    def isTarget(self, image_mask, circle):
        # image_mask: 原图掩码如image_target, image_circle经过滤色后的mask
        # circle: 霍夫圆或外接圆的信息(x, y, r)
        img_zero = np.zeros(np.shape(image_mask), dtype=np.uint8)
        # 空白图，准备绘制检测的圆
        enclosing_circle = img_zero.copy()
        # 都画实心圆…外接空心圆与圆环重合很少!
        enclosing_circle = cv2.circle(enclosing_circle, (circle[0], circle[1]), circle[2], (255, 255, 255), -1)
        img_base = enclosing_circle # 这里mask只有两个维度即xy，没有RGB维度
        # 借助检测的圆从原图掩码中抠出检测出的部分
        image_res = image_mask.copy()
        img_res = cv2.add(image_res, img_zero, mask=img_base)#添加掩码
        # 分别计算面积(像素和)
        cbase = (sum(sum(img_base.astype(int)/255)))
        ctarget = (sum(sum(img_res.astype(int)/255)))
        occupy_rate = float(ctarget)/float(cbase)
        # 设定阈值，占比大于阈值的应该是目标货物或圆环
        if occupy_rate>0.6:
            rospy.loginfo("[isTarget] occupy_rate=%f ", occupy_rate)
            return True
        # 其他模糊情况有待调试
        else:
            rospy.loginfo("[isTarget] occupy_rate=%f ", occupy_rate)
            return False
    
    # 尝试寻找轮廓和外接圆并绘制，与期望进行比对并返回可能的坐标
    def checkContours(self, image_mask, image_org, check_ball=True):
        # image_mask: 经过颜色过滤的掩码
        # image_org: 原图，应该可以是rgb或bgr的吧，主要是为了获取图像大小并输出标注
        # check_ball: 圆环检测主要是查漏，所以要
        binaries, contours, hierarchy = cv2.findContours(image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        circles = []
        if len(contours)>0:
            img_zero = np.zeros(np.shape(image_org), dtype=np.uint8)
            image_org_res = cv2.add(image_org, img_zero, mask=image_mask)#添加掩码
            dst = image_org_res.copy()
            dst = cv2.drawContours(dst, contours, -1, (125, 125, 125), 3)
            # 会不会有多个边缘?
            for contour in contours:
                (cx, cy), cr = cv2.minEnclosingCircle(contour)
                dst = cv2.circle(dst, (int(cx), int(cy)), int(cr), (201, 172, 221), 2)
                # 检测是否为圆(可能需要比较不同颜色的占比?)
                circle = np.array([cx,cy,cr]).astype(int)
                # 若能检测到实心，需要检测球则成功，否则检测圆环失败
                if self.isTarget(image_mask=image_mask, circle=circle):
                    if check_ball:
                        circles.append(circle)
                    # 实心的不是圆环
                    else:
                        continue
                # 若无实心，应该不是球，检测球的失败了
                elif check_ball:
                    continue
                # 若是检测圆环，则应该是圆环了
                else:
                    circles.append(circle)
        else:
            return None
        self.image_forward = dst
        # self.image_circle_pub.publish(self.bridge_.cv2_to_imgmsg(dst,encoding='bgr8'))
        # 找不到合适的边界或检测(圆环)失败
        if len(circles)>0:
            return circles
        else:
            return None
    
    # 检测圆环位置并计算导航信息
    def detectCircle(self):    # 返回圆环半径(图像)
        # 图像出错，返回异常值-1
        if self.image_ is None:
            return -1
        image_circle = self.image_.copy()
        #处理前视相机图像，检测圆环(也可能是货物)，这里先利用了黄色过滤
        image_circle_rgb = cv2.cvtColor(image_circle, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image_circle_rgb, cv2.COLOR_RGB2HSV)
        color_range = self.color_dict['y'][0]
        mask = cv2.inRange(image_hsv, color_range["lower"], color_range["upper"])
        # 霍夫圆检测
        # image_gray = cv2.cvtColor(image_circle, cv2.COLOR_BGR2GRAY)
        image_g = cv2.GaussianBlur(mask, (3, 3), 0)#滤波
        circles = cv2.HoughCircles(image_g, cv2.HOUGH_GRADIENT, 1, 100, param1=80, param2=40, minRadius=20, maxRadius=150)  #霍夫圆检测
        # 如果霍夫圆检测成功，则将数据加入队列；否则队列为空、circles为None，isHough置为False
        queue = deque()
        isHough = True
        if circles is not None:
            for i in circles[0,:]:
                queue.append(i)
        # 该循环至多执行两次: 第一次是前面检测出了霍夫圆; 第二次是这些圆都不是合适的圆环. 或一次: 未检测出霍夫圆, 直接用外接圆
        while isHough:
            # 队列为空要么是第一次没找到霍夫圆，要么是第一次找到后依次检查都被排除pop了，没有符合条件的(否则pop后还会append)
            if len(queue)==0:
                # 取消循环
                isHough = False
                # # 找不到霍夫圆可能是机身(上下左右)偏离过大了; 可能正确的圆环部分未进入视野, 所以尝试用外接圆寻找**有缺陷的圆环**
                # circles = self.checkContours(image_mask=mask, image_org=image_circle, check_ball=False)
            # 检查潜在的圆环(可能找到了一些被遮挡的圆)
            if circles is not None:
                if isHough:
                    circles = circles[0,:]
                circles = np.array(circles).astype(int)
                # 保留最大半径圆的信息以便导航 (2位设置最小半径=30;可能需要调整)
                cinfo = np.array([0, 0, 30]).astype(int)
                for i in circles:
                    # 假如是第一次检测出霍夫圆, 则每次首先pop掉一组圆的信息(从左侧,因为符合要求的会从右侧补上)
                    if len(queue)>0:
                        rospy.loginfo("[detectCircle] %d circle(s) in the queue now.", len(queue))
                        _ = queue.popleft() # i=queue.popleft()
                    # 确认是否为圆环(正常情况下应该不需要? 因为机身较高看不到货物)
                    if not self.isTarget(image_mask=mask, circle=i):
                        cv2.circle(image_circle, (i[0], i[1]), i[2], (125, 125, 125), 2)    # 画圆
                        cv2.circle(image_circle, (i[0], i[1]), 2, (125, 125, 125), 2)       # 画圆心
                        y = (i[0] - self.camera_info[0]) / i[2] * 35    # 图像x轴(左-右+), 实际y方向(左-右+)
                        z = -(i[1] - self.camera_info[1]) / i[2] * 35   # 图像y轴(下+上-), 实际z方向(下-上+)
                        #输出圆心的图像坐标和半径; 输出相对距离
                        rospy.loginfo("( %d , %d ), r= %d; y_LR=%f, z_DU=%f", i[0], i[1], i[2], y, z)
                        # 如果左右距离过远还是需要排除…
                        if math.fabs(y)>120:
                            rospy.loginfo("Too far! right or left")
                        # 如果上下距离过远,是不是没起飞? 或者上升不够?
                        if math.fabs(z)>60:
                            rospy.loginfo("Too high/low! up or down")
                            # 还没穿过一个圆环, 说明刚才可能起飞失败了! 返回异常值-3
                            self.navigating_queue_.append(['z', int(z)])
                            if self.ring_num_==0:
                                return -3
                            # 其他情况另作考虑，比如刚经过圆环2或4、检测货物后没飞起来?
                            else:
                                pass
                        # 上下左右距离没啥问题, 再看半径是否符合要求(前后距离)
                        elif i[2]>cinfo[2]:
                            cinfo = np.array([y, z, i[2]]).astype(int)
                            # 如果是在检测霍夫圆, 那么可以把这个信息放回队列了
                            if isHough:
                                rospy.loginfo("queue++, detected an underlying circle: "+str(cinfo))
                                queue.append(cinfo)
                        # 过滤掉其他距离较远的圆环
                        else:
                            rospy.loginfo("Too far!(forward) is that the next circle?")
                    # 如果检测发现是货物,其实可以直接到detectTarget了(×) 但最好想清楚以防重复检测
                    else:
                        rospy.loginfo("Oh no, this might be a ball! (color=\'y\')")
                # 假如是霍夫圆检测而且都无效, 则直接下一步(外接圆)
                if isHough and len(queue)==0:
                    continue
                # 非霍夫圆或者队列还有信息则意味着能检测出圆环(或毫无办法了), 总之可跳出循环
                else:
                    isHough = False
                # cinfo无效, 无法检测出合适的圆环, 直接跳出循环
                if cinfo[0]==0:
                    break
                cv2.circle(image_circle, (cinfo[0], cinfo[1]), cinfo[2], (0, 0, 255), 4)    # 画圆
                cv2.circle(image_circle, (cinfo[0], cinfo[1]), 2, (0, 0, 255), 3)           # 画圆心
                self.image_forward = image_circle
                # self.image_circle_pub.publish(self.bridge_.cv2_to_imgmsg(image_circle,encoding='bgr8'))
                y = cinfo[0]
                z = cinfo[1]
                rospy.loginfo("Circle: (LR, DU) = (dy, dz) = ( %d , %d )(move), r = %d (image)", y, z, cinfo[2])
                self.navigating_queue_.append(['y', y])
                self.navigating_queue_.append(['z', z])
                return cinfo[2]
            # 未检测到圆环，返回异常值-2; 可能的情况: 转弯失败或转弯后左右方向偏离较大
            else:
                return -2
        # 未检测到合适的圆环，返回异常值-4
        return -4
    
    # 判断是否检测到货物
    def detectTarget(self):
        self.Fly(['r', 0])#stop
        # 版本2.0: 一次性判断，无坐标追踪&反复确认
        if self.image_ is None:
            return False
        image_target = self.image_.copy()
        # 处理前视相机图像，检测货物
        image_target_rgb = cv2.cvtColor(image_target, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image_target_rgb, cv2.COLOR_RGB2HSV)
        # edges = cv2.Canny(image_target, 100, 200) # Canny边缘检测
        # 由于货物边缘不如圆环清晰，所以先尝试锁定颜色
        target = ''
        for color in self.color_dict.keys():
            for color_range in self.color_dict[color]:
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
                        # 输出圆心的图像坐标和半径
                        rospy.loginfo("( %d , %d ), r= %d", i[0], i[1], i[2])
                        self.image_circle_pub.publish(self.bridge_.cv2_to_imgmsg(image_target,encoding='bgr8'))
                        if self.isTarget(image_mask=mask, circle=i):
                            if len(target)==1 and target=='r' and color=='r':
                                rospy.loginfo("[detectTarget] Hough-Detected 2 red balls")
                            target += color
                            break
                        # # 内接正方形像素点求和取平均，查看是否为圆环
                        # len_a = round(i[2]/math.sqrt(2))-1
                        # inscribed_square = mask[(i[0]-len_a):(i[0]+len_a), (i[1]-len_a):(i[1]+len_a)]
                        # occupy_rate = sum(sum((inscribed_square.astype(int))/255))/inscribed_square.size
                        # # 设定阈值，占比大于阈值的是球而非圆环
                        # if occupy_rate>0.7:
                        #     rospy.logwarn("Detected ball with color \'%s\'~", color)
                else:
                    circles = self.checkContours(image_mask=mask, image_org=image_target, check_ball=True)
                    if circles is not None:
                        if len(circles)>1:
                            rospy.loginfo("[detectTarget] Contours-Detected %d balls with color: \'%s\'", len(circles), color)
                        target += color
        # 若检测到了货物，则发布识别到的货物信号,例如识别到了红色和黄色小球，则发布“ry”
        if len(target)==1:
            rospy.logwarn("Detected ball with color \'%s\'~", target)
            if self.ring_num_==1:
                self.target_result_ = target
            if self.ring_num_==4:
                self.target_result_ += target
            return True
        elif len(target)>0:
            rospy.loginfo("Too many circles? target="+target)
        # 若没有检测到货物，则返回False
        else:
            return False
    
    #飞行函数
    def Fly(self, next_nav):
        # 根据导航信息发布无人机控制命令
        magnitude = next_nav[1]
        if magnitude==0:
            self.publishCommand('stop')
            return
        direction = self.nav_dict[next_nav[0]][(magnitude>0)*1]
        magnitude = round(math.fabs(magnitude))
        rospy.loginfo("Fly: "+str(direction)+" "+str(int(magnitude)))
        # 最大移动尺度
        dmove = self.max_dis_dict[next_nav[0]]
        while magnitude>dmove:
            # t_begin = rospy.Time.now().to_sec()
            # 尽量走直角路线，每次都要校准yaw
            if next_nav[0]!='r':
                self.Adjust()
            # rospy.loginfo("dt_adjust="+str(rospy.Time.now().to_sec()-t_begin))
            # t_begin = rospy.Time.now().to_sec()
            self.publishCommand(direction+str(int(dmove)))
            magnitude -= dmove
            rospy.sleep(3)#max(1.5, float(dmove/10))
            # rospy.loginfo("dt_dmove="+str(rospy.Time.now().to_sec()-t_begin))
        # 最小移动尺度
        dmove = self.min_dis_dict[next_nav[0]]
        if next_nav[0]!='r':
            self.Adjust()
        if magnitude<dmove:
            self.publishCommand(self.nav_dict[next_nav[0]][(magnitude<0)*1]\
                                + str(int(2*dmove)))
            rospy.sleep(3)#max(1.5, float(2*dmove/10))
            self.publishCommand(direction+str(int(2*dmove+magnitude)))
            rospy.sleep(3)#max(1.5, float((2*dmove+magnitude)/10))
        else:
            self.publishCommand(direction+str(int(magnitude)))
        rospy.sleep(3)#max(1.5, float(magnitude/10/2))
    
    def Adjust(self):
        adjusted = False
        while not adjusted:
            (yaw, pitch, roll) = self.R_wu_.as_euler('zyx', degrees=True)
            # 最近的直角角度为round(yaw/90)*90, 不过这里由变量self.theta记录并由穿过圆环的个数检测更新
            # 参见 nav_dict 变量定义注释（顺时针为正）
            #   仿真环境中 x 正方向 yaw=0，y正方向（左侧）yaw=90
            #   若令 yaw+=yaw_diff，则 yaw_diff>0 应该是逆时针ccw，取反则是顺时针cw，和定义匹配
            yaw_diff = -(self.theta - yaw) # 即: yaw-self.theta
            if math.fabs(yaw_diff)>180:
                yaw_diff = (360 - yaw_diff) if yaw_diff>0 else (360 + yaw_diff)
            if math.fabs(yaw_diff)>10:
                direction = self.nav_dict['r'][(yaw_diff>0)*1]
                yaw_diff = round(math.fabs(yaw_diff))
                # yaw_diff = int(max(yaw_diff, self.min_dis_dict['r'])) # 最小移动尺度
                rospy.loginfo("Adjust yaw (from yaw="+str(yaw)+" to dest="+str(self.theta)+" by \'"+direction+str(int(yaw_diff))+"\'")
                self.publishCommand(direction+str(int(yaw_diff)))
                rospy.sleep(max(1.5, float(round(yaw_diff/10))))
            else:
                adjusted = True
        # 悬停的同时应该可以调整好pitch和roll
        self.publishCommand('stop')
        rospy.sleep(0.2)
    
    # 接收前视相机图像
    def imageCallback(self, msg):
        try:
            self.image_ = self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
            if self.image_forward is not None:
                self.image_circle_pub.publish(self.bridge_.cv2_to_imgmsg(self.image_forward,encoding='bgr8'))
        except CvBridgeError as err:
            print(err)
    # 接受下视相机图像
    def imageCallback_down(self,msg):
        try:
            self.image_down_ = self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
            if self.image_down is not None:
                self.image_tag_pub.publish(self.bridge_.cv2_to_imgmsg(self.image_down,encoding='bgr8'))
        except CvBridgeError as err:
            print(err)
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
    # 接收开始信号
    def startcommandCallback(self, msg):
        self.is_begin_ = msg.data
    # 接收相机内参
    def camera_info_callback(self, msg):
        self.camera_info = [msg.K[2], msg.K[5]]

if __name__ == '__main__':
    cn = TestNode()