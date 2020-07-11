# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import math

# 固定尺寸
def resizeImg(image, height=900):
    h, w = image.shape[:2]
    pro = height / h
    size = (int(w * pro), int(height))
    img = cv2.resize(image, size)

    return img


# 边缘检测
def getCanny(image):
    # 高斯模糊
    binary = cv2.GaussianBlur(image, (1, 1), 1, 1)
    # 边缘检测
    binary = cv2.Canny(binary, 60, 240, apertureSize=3)
    # 膨胀操作，尽量使边缘闭合
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    # 查看膨胀后的边缘图
    # cv2.imshow('getCanny',binary)
    # cv2.waitKey(0)
    return binary


# 求出面积最大的轮廓
def findMaxContour(image):
    # 寻找边缘
    _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 计算面积
    max_area = 0.0
    max_contour = []

    second_area=0.0
    second_countour = []

    for contour in contours:
        currentArea = cv2.contourArea(contour)
        if currentArea > max_area:
            max_area = currentArea
            max_contour = contour
            second_countour =max_contour
            second_area=max_area

    #print(max_area)
    #print(second_area)
    return max_contour, max_area,second_countour


# 求出面积大于百分比的所有轮廓
def findProposalCountour(image):
  # 寻找边缘
   # _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # 计算面积

    proposalContours = []
    sp = image.shape
 
    for contour in contours:
        currentArea = cv2.contourArea(contour)
        if cv2.contourArea(contour) >= 0.2*sp[0]*sp[1] :
            # print(currentArea)
            # print(0.1*sp[0]*sp[1])
            #print('wjytetstetset')
            # vertices = np.array([contour], dtype=np.int32)
            # cv2.polylines(image,vertices,True,(255 ,255, 0),10) 
            # cv2.imshow('boxes',image)
            # cv2.waitKey(0)
            proposalContours.append(contour)
 
    #print(proposalContours)

    return proposalContours  


# 多边形拟合凸包的四个顶点
def getBoxPoint(contour):
    # 多边形拟合凸包
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    approx = approx.reshape((len(approx), 2))
    return approx

#计算三个点的角度，最后一个为重合点
def getLineAngle(point_1, point_3, point_2):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
    #A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    #C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))

    return B

#计算四边形的最大内角
def getMaxAngle(box):
    #左上
    lefttopA = getLineAngle(box[1],box[3],box[0])
    maxAngle = lefttopA
    #右上
    righttopA = getLineAngle(box[0],box[2],box[1])
    #右下
    rightdownA = getLineAngle(box[1],box[3],box[2])
    #右上
    leftdwonA = getLineAngle(box[2],box[0],box[3])
    # print('getLineAngle')
    # print('lefttopA')
    # print(lefttopA)
    # print('righttopA')
    # print(righttopA)
    # print('rightdownA')
    # print(rightdownA)
    # print('leftdwonA')
    # print(leftdwonA)
    # print('getLineAngleEnd')
    if righttopA > maxAngle:
        maxAngle = righttopA
    if rightdownA > maxAngle:
        maxAngle = rightdownA
    if leftdwonA > maxAngle:
        maxAngle = leftdwonA
    # print(maxAngle)
    return maxAngle



# 适配原四边形点集
def adaPoint(box, pro):
    box_pro = box
    if pro != 1.0:
        box_pro = box/pro
    box_pro = np.trunc(box_pro)
    return box_pro


# 四边形顶点排序，[top-left, top-right, bottom-right, bottom-left]
def orderPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 计算长宽
def pointDistance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))


# 透视变换
def warpImage(image, box):
    w, h = pointDistance(box[0], box[1]), \
           pointDistance(box[1], box[2])
    dst_rect = np.array([[0, 0],
                         [w - 1, 0],
                         [w - 1, h - 1],
                         [0, h - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(box, dst_rect)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

def enLargeImgHorizon(boxes):
    Scaling = 0.3
    #bottomf = 0.3

    # print(boxes)

    #如果第一个顶点比第二个顶点高
    if boxes[0][1] <= boxes[1][1]:
        # 左上角的新坐标
        topleftx = boxes[0][0] - (boxes[1][0] - boxes[0][0]) * Scaling
        toplefty = boxes[0][1] - (boxes[1][1] - boxes[0][1]) * Scaling
        # 右上角的新坐标
        toprightx = boxes[1][0] + (boxes[1][0] - boxes[0][0]) * Scaling
        toprighty = boxes[1][1] + (boxes[1][1] - boxes[0][1]) * Scaling
    else:
        # 如果第一个顶点比第二个顶点低
        # 左上角的新坐标
        topleftx = boxes[0][0] - (boxes[1][0] - boxes[0][0]) * Scaling
        toplefty = boxes[0][1] + (boxes[0][1] - boxes[1][1]) * Scaling
        # 右上角的新坐标
        toprightx = boxes[1][0] + (boxes[1][0] - boxes[0][0]) * Scaling
        toprighty = boxes[1][1] - (boxes[0][1] - boxes[1][1]) * Scaling

        # 如果第四个顶点比第三个顶点高
    if boxes[3][1] <= boxes[2][1]:
        # 左下角的新坐标
        downleftx = boxes[3][0] - (boxes[2][0] - boxes[3][0]) * Scaling
        downlefty = boxes[3][1] - (boxes[2][1] - boxes[3][1]) * Scaling
        # 右下角的新坐标
        downrightx = boxes[2][0] + (boxes[2][0] - boxes[3][0]) * Scaling
        downrighty = boxes[2][1] + (boxes[2][1] - boxes[3][1]) * Scaling
    else:
        # 如果第四个顶点比第三个顶点低
        # 左上角的新坐标
        downleftx = boxes[3][0] - (boxes[2][0] - boxes[3][0]) * Scaling
        downlefty = boxes[3][1] + (boxes[2][1] - boxes[3][1]) * Scaling
        # 右上角的新坐标
        downrightx = boxes[2][0] + (boxes[2][0] - boxes[3][0]) * Scaling
        downrighty = boxes[2][1] - (boxes[3][1] - boxes[2][1]) * Scaling

        # 重新赋值4个顶点坐标
    boxes[0][0] = topleftx
    boxes[0][1] = toplefty

    boxes[1][0] = toprightx
    boxes[1][1] = toprighty

    boxes[2][0] = downrightx
    boxes[2][1] = downrighty

    boxes[3][0] = downleftx
    boxes[3][1] = downlefty
    #print(boxes)
    return boxes


#计算延长线的点坐标
def x1y1(x1, y1, x2, y2, rate):
    Up_Y = rate*(y1-y2) + y2
    Up_X = rate*(x1-x2) + x2
    Down_Y = y1 - rate*(y1-y2)
    Down_X = x1 - rate*(x1-x2)
    return Up_X, Up_Y, Down_X, Down_Y

#全部延长线的点坐标
def opt(a_x, a_y, b_x, b_y, c_x, c_y, d_x, d_y):
    #print(a_x, a_y ,b_x ,b_y, c_x, c_y, d_x, d_y)
    a = int((a_y - d_y)*(a_y - d_y)/10000 + (a_x - d_x)*(a_x - d_x)/10000)
    b = int((a_x - b_x)*(a_x - b_x)/10000 + (a_y - b_y)*(a_y - b_y)/10000)
    #print(a) #纵向
    #print(b) #横向
    if a > b:
        x_2, y_2, x_1, y_1 = x1y1(b_x, b_y, a_x, a_y, 1.1) #2, 1  ab
        x_4, y_4, x_3, y_3 = x1y1(c_x, c_y, d_x, d_y, 1.1) #4, 3  cd

        x_6, y_6, x_5, y_5 = x1y1(c_x, c_y, b_x, b_y, 1.3) #6, 5  bc
        x_8, y_8, x_7, y_7 = x1y1(d_x, d_y, a_x, a_y, 1.3) #8, 7  ad
        print("high")
    else:
        x_2, y_2, x_1, y_1 = x1y1(b_x, b_y, a_x, a_y, 1.3) #2, 1  ab
        x_4, y_4, x_3, y_3 = x1y1(c_x, c_y, d_x, d_y, 1.3) #4, 3  cd

        x_6, y_6, x_5, y_5 = x1y1(c_x, c_y, b_x, b_y, 1.1) #6, 5  bc
        x_8, y_8, x_7, y_7 = x1y1(d_x, d_y, a_x, a_y, 1.1) #8, 7  ad
        print("wide")

    return x_1, y_1, x_2, y_2,x_3, y_3, x_4, y_4, x_5, y_5, x_6, y_6, x_7, y_7, x_8, y_8

def find_max(a):
    b = []
    max_1 = 0
    max_2 = 0
    for i in range(4):
        if a[i][1] >= a[(i+5)%4][1] and a[i][1] >= a[(i+6)%4][1] and a[i][1] >= a[(i+7)%4][1]:
            max_1 = a[i][1]
            max_i = i
    for i in range(4):
        if i != max_i:
            b.append(i)
    for i in b:
            if a[i][1] >= a[b[0]][1] and a[i][1] >= a[b[1]][1] and a[i][1] >= a[b[2]][1]:
                max_2 = a[i][1]
                max_j = i
    #print(max_i, max_j)
    if max_i<max_j:
        min = max_i
    else:
        min = max_j
    shut = a[min][0]
    
    for i in range(4):
        temp_x, temp_y = a[0][0], a[0][1]
        for j in range(3):
            a[j][0], a[j][1] = a[j+1][0], a[j+1][1]
        a[3][0], a[3][1] = temp_x, temp_y
        if shut == a[0][0]:
            break

    return a

#四边形扩边 30%
def enLargeImg(boxes):
    #print('begain')
    #print(boxes)
    boxes = find_max(boxes)
    #print(boxes)
    DATA = []
    for i in range(4):
        for j in range(2):
            DATA.append(boxes[i][j])
    for i in range(4):
        for j in range(2):
            DATA.append(boxes[i][j])
    data = []
    for i in DATA:
        data.append(i)
    #print(data)
    #print(opt(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]))
    x_1, y_1, x_2, y_2,x_3, y_3, x_4, y_4, x_5, y_5, x_6, y_6, x_7, y_7, x_8, y_8 = opt(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])
    
    A_X = (y_7- y_1 + (y_3-y_1)/(x_3-x_1)*x_1 - (y_5-y_7)/(x_5-x_7)*x_7)/((y_3-y_1)/(x_3-x_1) - (y_5-y_7)/(x_5-x_7))
    A_Y = (y_5-y_7)*(A_X-x_7)/(x_5-x_7)+y_7
    boxes[0][0] = A_X
    boxes[0][1] = A_Y

    B_X = (y_7- y_2 + (y_4-y_2)/(x_4-x_2)*x_2 - (y_5-y_7)/(x_5-x_7)*x_7)/((y_4-y_2)/(x_4-x_2) - (y_5-y_7)/(x_5-x_7))
    B_Y = (y_5-y_7)*(B_X-x_7)/(x_5-x_7)+y_7
    boxes[1][0] = B_X
    boxes[1][1] = B_Y

    C_X = (y_8- y_2 + (y_4-y_2)/(x_4-x_2)*x_2 - (y_6-y_8)/(x_6-x_8)*x_8)/((y_4-y_2)/(x_4-x_2) - (y_6-y_8)/(x_6-x_8))
    C_Y = (y_4-y_2)*(C_X-x_2)/(x_4-x_2)+y_2
    boxes[2][0] = C_X
    boxes[2][1] = C_Y

    D_X = (y_8- y_1 + (y_3-y_1)/(x_3-x_1)*x_1 - (y_6-y_8)/(x_6-x_8)*x_8)/((y_3-y_1)/(x_3-x_1) - (y_6-y_8)/(x_6-x_8))
    D_Y = (y_6-y_8)*(D_X-x_8)/(x_6-x_8)+y_8
    boxes[3][0] = D_X
    boxes[3][1] = D_Y
    
    return boxes

def process(cv2_img):
    # image = cv2.imread(image_path)
    ratio = 900 / cv2_img.shape[0]
    img = resizeImg(cv2_img)
    binary_img = getCanny(img)
    proposalContours = findProposalCountour(binary_img)
    i = 1
    maxboxarea = 0.0
    maxbox_1 = [0, 0]
    maxbox_2 = [0, 0]
    box_2 = []
    for contour in proposalContours:
        box = getBoxPoint(contour)
        #print(box)
        ageltmp = getMaxAngle(box)
        #print(ageltmp)
        if  ageltmp <= 93:
            curArea = cv2.contourArea(box)
            vertices = np.array([box], dtype=np.int32)
            cv2.polylines(img,vertices,True,(255,0,0)) 
            cv2.imshow('boxes',img)
            cv2.waitKey(0)
            if curArea > maxboxarea:
                maxboxarea = curArea
                maxbox_1 = box
    #print(maxbox_1)
    #print(len(maxbox_1))
    if len(maxbox_1) == 2:
        return 1, 0
    maxbox = find_max(maxbox_1)
    #print(maxbox)
    #print("hahahaha")

    a1_x, a1_y, d1_x, d1_y = maxbox_1[0][0], maxbox_1[0][1], maxbox_1[3][0], maxbox_1[3][1]
    #print(a1_x, a1_y, d1_x, d1_y)
    a = (a1_y-d1_y)/(a1_x-d1_x)
    b = (d1_x-a1_x)/(a1_y-d1_y)

    for contour in proposalContours:
        box = getBoxPoint(contour)
        #print(box)
        if box[0][0] != maxbox_1[0][0]:
            #print(box)
            #print(x_new, y_new)
            ageltmp = getMaxAngle(box)
            #print(ageltmp)
            if  ageltmp <= 93:
                curArea = cv2.contourArea(box)
                vertices = np.array([box], dtype=np.int32)
                cv2.polylines(img,vertices,True,(255,0,0)) 
                cv2.imshow('boxes',img)
                cv2.waitKey(0)

                maxbox_2 = find_max(box)
                #print('ha')
                #print(maxbox_2)
                x_new, y_new = box[0][0], box[0][1]
                #print(x_new, y_new)
                x_miss = (y_new-a1_y+a*a1_x-b*x_new)/(a-b)
                y_miss = a*(x_miss-a1_x)+a1_y
                #print(x_miss, y_miss)

                reuslt = int((x_miss - x_new)*(x_miss - x_new) + (y_miss + y_new)*(y_miss + y_new))
                #print(reuslt)
                if reuslt/1000 < 10000:
                    #print('in?')
                    #print(maxbox_1[0][1], maxbox_2[0][1])
                    #print(maxbox_1[0][1]>maxbox_2[0][1])
                    if (maxbox_1[0][1] > maxbox_2[0][1]):
                        maxbox = maxbox_1
                        maxbox[2] = maxbox_2[2]
                        maxbox[3] = maxbox_2[3]
                    else:
                        maxbox = maxbox_2
                        maxbox[2] = maxbox_1[2]
                        maxbox[3] = maxbox_1[3]      
    maxbox = find_max(maxbox)
    #print('he')
    print(maxbox)
    warped = cv2_img

    if len(maxbox):
        #print('haha')
        boxes = orderPoints(maxbox) 
        #查看排序后的框图
        #print(boxes)
        # vertices = np.array([boxes], dtype=np.int32)
        # cv2.polylines(img,vertices,True,(255,0,0)) 
        # cv2.imshow('boxes',img)
        # cv2.waitKey(0)
        boxes = adaPoint(boxes, ratio)
        boxes = orderPoints(boxes)
        sp = cv2_img.shape
        #print(sp[0]*sp[1])
        #print(cv2.contourArea(boxes))

        #如果处理后的外框，整体面积大于22%，才进行行透视变换处理
        #if cv2.contourArea(boxes) >= 0.215*sp[0]*sp[1]: #正式0.215
        if cv2.contourArea(boxes) >= 0.2*sp[0]*sp[1]:
            #print('one')
            #print(boxes)
            boxes = enLargeImg(boxes)
            #查看缩放后的框图
            #print('two')
            #print(boxes)
            vertices = np.array([boxes], dtype=np.int32)
            cv2.polylines(cv2_img,vertices,True,(255,0,0)) 
            cv2.imshow('boxes',cv2_img)
            cv2.waitKey(0)

            warped = warpImage(cv2_img, boxes) # 透视变化
            #print(1)
            #warpImage(image, boxes)

    # save the transformed image
    return warped, 1

def cv_imread(file_path):           #为了方便，把它又定义成了了一个函数，方便调用
    cv_img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img
def img_resize(filepath):
    sum = 0
    for parent, dirnames, filenames in os.walk(filepath):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字  
        for filename in filenames:
            sum = sum+1
            img_path =os.path.join(parent,filename)   #将父目录和名字组合到一起形成访问路径
            savepaths=os.path.join('F:/desktop/桌面/log',filename)         
            img=cv_imread(img_path) # 读取图片  
            print('第',sum,'张：')
            img, lock = process(img)
            if lock == 1:
                cv2.imencode('.jpg',img)[1].tofile(savepaths)
            else:
                print('no output')


if __name__ == '__main__':
    
    #单个
    cv2_img = cv2.imread('test_image.png')
    cv2_img, lock = process(cv2_img)
    if lock == 1:
        cv2.imwrite("result.png", cv2_img)
    else:
        print('no output')
    
    '''
    #批量
    img_resize('F:/desktop/桌面/test')
    '''