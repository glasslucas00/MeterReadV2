'''
 ┌─────────────────────────────────────────────────────────────┐
 │┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
 ││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
 │├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
 ││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
 │├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
 ││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
 │├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
 ││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
 │└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
 │      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
 │      └───┴─────┴───────────────────────┴─────┴───┘          │
 └─────────────────────────────────────────────────────────────┘

Author: lucas
Date: 2022-05-13 00:03:00
LastEditTime: 2022-05-13 16:30:57
LastEditors: lucas
Description: 仪表识别核心
FilePath: \meter_without_gui\MeterReadV2\MeterClass.py
CSDN:https://blog.csdn.net/qq_27545821?spm=1000.2115.3001.5343
github: https://github.com/glasslucas00?tab=repositories
'''
from math import sqrt
import cv2
import numpy as np
import os
import random 
import glob

class Functions:
    @staticmethod
    def GetClockAngle(v1, v2): 
         # 2个向量模的乘积 ,返回夹角
        TheNorm = np.linalg.norm(v1)*np.linalg.norm(v2)
        # 叉乘
        rho = np.rad2deg(np.arcsin(np.cross(v1, v2)/TheNorm))
        # 点乘
        theta = np.rad2deg(np.arccos(np.dot(v1,v2)/TheNorm))
        if rho > 0:
            return  360-theta
        else:
            return theta
    @staticmethod
    def Disttances(a, b):
        x1, y1 = a
        x2, y2 = b
        Disttances = int(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        return Disttances

    @staticmethod
    def couputeMean(deg):
        """
        :funtion :
        :param b:
        :param c:
        :return:
        """
        if (True):
            # new_nums = list(set(deg)) #剔除重复元素
            mean = np.mean(deg)
            var = np.var(deg)
            # print("原始数据共", len(deg), "个\n", deg)
            '''
            for i in range(len(deg)):
                print(deg[i],'→',(deg[i] - mean)/var)
                #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据
            '''
            # print("中位数:",np.median(deg))
            percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
            # print("分位数：", percentile)
            # 以下为箱线图的五个特征值
            Q1 = percentile[0]  # 上四分位数
            Q3 = percentile[2]  # 下四分位数
            IQR = Q3 - Q1  # 四分位距
            ulim = Q3 + 2.5 * IQR  # 上限 非异常范围内的最大值
            llim = Q1 - 1.5 * IQR  # 下限 非异常范围内的最小值

            new_deg = []
            uplim = []
            for i in range(len(deg)):
                if (llim < deg[i] and deg[i] < ulim):
                    new_deg.append(deg[i])
            # print("清洗后数据共", len(new_deg), "个\n", new_deg)
        new_deg = np.mean(new_deg)

        return new_deg


class MeterDetection:
    def __init__(self,path):
        self.imageName=path.split('/')[-1].split('.')[0]
        self.outputPath=os.getcwd()+'/outputs/'
        self.image=cv2.imread(path)
        self.circleimg=None
        self.panMask=None           #霍夫圆检测切割的表盘图片
        self.poniterMask =None      #指针图片
        self.numLineMask=None       #刻度线图片
        self.centerPoint=None       #中心点[x,y]
        self.farPoint=None         #指针端点[x,y]
        self.zeroPoint=None         #起始点[x,y]
        self.r=None                 #半径
        self.divisionValue=100/360
        self.makeFiledir()
        self.markZeroPoint()


    def makeFiledir(self):
        """ 创建输出文件夹"""
        if not os.path.exists(self.outputPath):  # 是否存在这个文件夹
            os.makedirs(self.outputPath)  # 如果没有这个文件夹，那就创建一个
    
    def markZeroPoint(self):
        img =self.image
        def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                p0 = [x, y]
                self.zeroPoint=p0
                cv2.circle(img, (x, y), 2, (120, 0, 255), thickness=-1)
                cv2.imshow("image", img)
                

            elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键fang
                cv2.waitKey(500)
                cv2.destroyWindow("image")
                
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
        cv2.imshow('image', img)
        cv2.waitKey()
           
    def ImgCutCircle(self):
        """
        :param pyrMeanShiftFiltering(input, 10, 100) 均值滤波
        :param 霍夫概率圆检测
        :param mask操作提取圆
        :return: 半径，圆心位置

        """
        img=self.image
        dst = cv2.pyrMeanShiftFiltering(img, 10, 100)
        cimage = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 80, param1=100, param2=20, minRadius=80, maxRadius=0)
        circles = np.uint16(np.around(circles))  # 把类型换成整数
        r_1 = circles[0, 0, 2]
        c_x = circles[0, 0, 0]
        c_y = circles[0, 0, 1]
        circle = np.ones(img.shape, dtype="uint8")
        circle = circle * 255
        cv2.circle(circle, (c_x, c_y), int(r_1), 0, -1)
        bitwiseOr = cv2.bitwise_or(img, circle)
        cv2.imwrite(self.outputPath+self.imageName + '_1_imgCutCircle.jpg' , bitwiseOr)
        self.cirleData = [r_1, c_x, c_y]
        self.panMask=bitwiseOr
       
        return bitwiseOr

    def ContoursFilter(self):
        """
        :funtion : 提取刻度线，指针
        :param a: 高斯滤波 GaussianBlur，自适应二值化adaptiveThreshold，闭运算
        :param b: 轮廓寻找 findContours，
        :return:lineSet,new_needleset
        """
        r_1, c_x, c_y = self.cirleData

        img = self.image.copy()
        # cv2.circle(img, (c_x, c_y), 20, (23, 28, 28), -1)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(~gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
        # cv2.circle(binary, (c_x, c_y), int(r_1*0.5), (0, 0, 0),5)
        # 闭运算
        # kernel = np.ones((3, 3), np.uint8)
        #膨胀
        # dilation = cv2.dilate(binary, kernel, iterations=1)
        # kernel2 = np.ones((3, 3), np.uint8)
        #腐蚀
        # erosion = cv2.erode(dilation, kernel2, iterations=1)
        
        #轮廓查找，根据版本不同，返回参数不同
        if cv2.__version__ >'4.0.0':
            contours, hier = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            aa,contours, hier = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cntset = []  # 刻度线轮廓集合
        cntareas = []  # 刻度线面积集合

        needlecnt = []  # 指针轮廓集合
        needleareas = []  # 指针面积集合
        radiusLength = [r_1 * 0.6, r_1 * 1] # 半径范围

      
        # cv2.drawContours(img, contours, -1, (255, 90, 60), 2)
        # cv2.imwrite(self.outputPath+self.imageName + '_2_----numLineMask.jpg' , img)
        localtion = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            # print(rect)
            #（中心点坐标，（宽度，高度）,旋转的角度）=   = rect
            a, (w, h), c = rect  
            w = int(w)
            h = int(h)
            ''' 满足条件:“长宽比例”，“面积”'''
            if h == 0 or w == 0:
                pass
            else:
                dis = Functions.Disttances((c_x, c_y), a)
                # if (radiusLength[0] < dis and radiusLength[1] > dis):
                if (radiusLength[0] < dis and radiusLength[1] > dis):
                    #矩形筛选
                    if h / w > 4 or w / h > 4:
                        localtion.append(dis)
                        cntset.append(cnt)
                        cntareas.append(w * h)
                else:
                    if w > r_1 / 2 or h > r_1 / 2:
                        needlecnt.append(cnt)
                        needleareas.append(w * h)
        cntareas = np.array(cntareas)
        areasMean = Functions.couputeMean(cntareas)  # 中位数，上限区
        new_cntset = []
        # 面积
        for i, cnt in enumerate(cntset):
            if (cntareas[i] <= areasMean * 1.5 and cntareas[i] >= areasMean * 0.8):
                new_cntset.append(cnt)

        self.r = np.mean(localtion)
        mask = np.zeros(img.shape[0:2], np.uint8)
        self.poniterMask = cv2.drawContours(mask, needlecnt, -1, (255, 255, 255), -1)  # 生成掩膜
        mask = np.zeros(img.shape[0:2], np.uint8)
        self.numLineMask = cv2.drawContours(mask, new_cntset, -1, (255, 255, 255), -1)  # 生成掩膜

        cv2.imwrite(self.outputPath+self.imageName + '_2_numLineMask.jpg' , self.numLineMask)
        cv2.imwrite(self.outputPath+self.imageName + '_3_poniterMask.jpg' , self.poniterMask)
        # for cnt in needlecnt:
        #     cv2.fillConvexPoly(mask,cnt , 255)
        self.new_cntset=new_cntset
        
        return new_cntset
    
    def FitNumLine(self):
        """ 轮廓拟合直线"""
        lineSet = []  # 拟合线集合
        img=self.image.copy()
        for cnt in self.new_cntset:
            rect = cv2.minAreaRect(cnt)
            # 获取矩形四个顶点，浮点型
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.polylines(img, [box], True, (0, 255, 0), 1)  # pic
            output = cv2.fitLine(cnt, 2, 0, 0.001, 0.001)
            k = output[1] / output[0]
            k = round(k[0], 2)
            b = output[3] - k * output[2]
            b = round(b[0], 2)
            x1 = 1
            x2 = img.shape[0]
            y1 = int(k * x1 + b)
            y2 = int(k * x2 + b)
            # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            #lineSet:刻度线拟合直线数组，k斜率 b
            lineSet.append([k, b])  # 求中心点的点集[k,b]
        cv2.imwrite(self.outputPath+self.imageName + '_4_fitNumLine.jpg' , img)
        self.lineSet=lineSet
        return lineSet

    def getIntersectionPoints(self):
        img = self.image
        lineSet=self.lineSet
        w, h, c = img.shape
        point_list = []
        xlist=[]
        ylist=[]
        if len(lineSet) > 2:
            # print(len(lineSet))
            np.random.shuffle(lineSet)
            lkb = int(len(lineSet) / 2)
            kb1 = lineSet[0:lkb]
            kb2 = lineSet[lkb:(2 * lkb)]
            # print('len', len(kb1), len(kb2))
            kb1sample = random.sample(kb1, int(len(kb1) / 2))
            kb2sample = random.sample(kb2, int(len(kb2) / 2))
        else:
            kb1sample = lineSet[0]
            kb2sample = lineSet[1]
        for i, wx in enumerate(kb1sample):
            # for wy in kb2:
            for wy in kb2sample:
                k1, b1 = wx
                k2, b2 = wy
                # print('kkkbbbb',k1[0],b1[0],k2[0],b2[0])
                # k1-->[123]
                try:
                    if (b2 - b1) == 0:
                        b2 = b2 - 0.1
                    if (k1 - k2) == 0:
                        k1 = k1 - 0.1
                    x = (b2 - b1) / (k1 - k2)
                    y = k1 * x + b1
                    x = int(round(x))
                    y = int(round(y))
                except:
                    x = (b2 - b1 - 0.01) / (k1 - k2 + 0.01)
                    y = k1 * x + b1
                    x = int(round(x))
                    y = int(round(y))
                # x,y=solve_point(k1, b1, k2, b2)
                if x < 0 or y < 0 or x > w or y > h:
                    break
                # point_list.append([x, y])
                xlist.append(x)
                ylist.append(y)
                # cv2.circle(img, (x, y), 2, (122, 22, 0), 2)
        # print('point_list',point_list)
        cx=int(np.mean(xlist))
        cy=int(np.mean(ylist))
        self.centerPoint=[cx,cy]
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), 2)
        cv2.imwrite(self.outputPath+self.imageName + '_5_IntersectionPoints.jpg' , img)
        return img

    def FitPointerLine(self):
        img =self.poniterMask
        orgin_img=self.image.copy()
        # kernel = np.ones((3, 3), np.uint8)
        # mask = cv2.dilate(img, kernel, iterations=1)
        # img = cv2.erode(mask, kernel, iterations=1)
        lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength=int(self.r / 2), maxLineGap=2)
        # nmask = np.zeros(img.shape, np.uint8)
        # lines = mential.findline(self=0, cp=[x, y], lines=lines)
        # print('lens', len(lines))
        dmax=0
        pointerLine=[]
        #最长的线段为指针
        for line in lines:
            x1, y1, x2, y2 = line[0]
            d1=Functions.Disttances((x1, y1),(x2, y2))
            if(d1>dmax):
                dmax=d1
                pointerLine=line[0]      
        x1, y1, x2, y2 = pointerLine
        d1=Functions.Disttances((x1, y1),(self.centerPoint[0],self.centerPoint[1]))
        d2=Functions.Disttances((x2, y2),(self.centerPoint[0],self.centerPoint[1]))
        if d1 > d2:
            self.farPoint = [x1, y1]
        else:
            self.farPoint = [x2, y2]

        cv2.line(orgin_img, (x1, y1), (x2, y2), 20, 1, cv2.LINE_AA)
        cv2.circle(orgin_img,(self.farPoint[0],self.farPoint[1]), 2, (0, 0, 255),2)
        cv2.imwrite(self.outputPath+self.imageName + '_6_PointerLine.jpg' , img)
        cv2.imwrite(self.outputPath+self.imageName + '_7_PointerPoint.jpg' , orgin_img)
     
    def Readvalue(self):
        try:
            self.ImgCutCircle()
            self.ContoursFilter()
            self.FitNumLine()
            self.getIntersectionPoints()
            self.FitPointerLine()
            v1=[self.zeroPoint[0]-self.centerPoint[0],self.centerPoint[1]-self.zeroPoint[1]]
            v2=[self.farPoint[0]-self.centerPoint[0],self.centerPoint[1]-self.farPoint[1]]
            theta=Functions.GetClockAngle(v1,v2)
            readValue=self.divisionValue*theta
            print(theta,readValue)
            return readValue
        except Exception as e:# 写一个except
            print("程序错误：",e)

if __name__ =="__main__":  

    #多张图片，修改输入文件夹

    # imglist=glob.glob('input/*.jpg')  
    # for imgpath in  imglist: 
    #     A=MeterDetection(imgpath)
    #     A.Readvalue()
    #一张图片
    imgpath='2.jpg'
    A=MeterDetection(imgpath)
    readValue=A.Readvalue()
