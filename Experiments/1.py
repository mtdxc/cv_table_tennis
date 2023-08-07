
# 功能：将图像中某一四边形变换为矩阵

import cv2
import numpy as np
        
img = cv2.imread('1.png')
#获取源图像宽和高
w = img.shape[0]
h = img.shape[1]

Recpoints = []  #空列表用于储存原始图像四点坐标

def draw_circle(event,x,y,flags,param):
    global Recpoints
    frame = img
    if event==cv2.EVENT_LBUTTONDBLCLK:
        Recpoints.append([x,y])
        print(x,y)
        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
        cv2.imshow('image', frame)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image',draw_circle)

while(len(Recpoints) < 4):
    for (x, y) in Recpoints:
        cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
    cv2.imshow('image', img)
    key = cv2.waitKey() & 0xFF
    print(key)
    if key==27:  #按ESC退出
        break
    elif key == 110:
        Recpoints = []

if len(Recpoints) < 4:
    print('must mark 4 point to continue')
    exit(0)

#源图像中四边形坐标点（获取坐标点方法可参照我上篇博文）
point1 = np.array(Recpoints, dtype = "float32")
#转换后得到矩形的坐标点
point2 = np.array([[0,0],[320,0],[0,180],[320,180]],dtype = "float32")
# point2 = np.array([[0,180],[320,180],[0,0],[320,0]],dtype = "float32")
print('point1', point2)
print('point2', point2)
M = cv2.getPerspectiveTransform(point1,point2)
print('M', M)
out_img = cv2.warpPerspective(img,M,(320,180))#(w,h))
cv2.imshow("img", out_img)

def getMapPos(point):
    # 坐标映射矩阵
    pt = np.array((point[0], point[1], 1))
    pt2 = np.dot(M, pt)
    pt2 /= pt2[2]
    return pt2[0], pt2[1]

print("now check getMapPos with point1")
for pt in point1:
    print(pt, '=', getMapPos(pt))

cv2.waitKey(0)
cv2.destroyAllWindows()