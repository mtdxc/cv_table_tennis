
# 功能：将图像中某一四边形变换为矩阵

import cv2
import numpy as np
import sys
choosen_video = 'input_video.mp4'
if len(sys.argv) > 1:
    choosen_video = sys.argv[1]
# Begin tracking
cap = cv2.VideoCapture(choosen_video)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print('%s (%dx%d@%d)' % (choosen_video, width, height, fps))

ret, img = cap.read()
if not ret:
    print('unable to open file ', choosen_video)
    exit()

nPoint = 4
Recpoints = []  #空列表用于储存原始图像四点坐标
selname = 'please select %d point of rect' % nPoint
def draw_circle(event,x,y,flags,param):
    global Recpoints
    frame = img
    if event==cv2.EVENT_LBUTTONDBLCLK:
        Recpoints.append([x,y])
        print(x,y)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow(selname, frame)

cv2.namedWindow(selname,cv2.WINDOW_NORMAL)
cv2.setMouseCallback(selname, draw_circle)

while(len(Recpoints) < nPoint):
    for (x, y) in Recpoints:
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
    cv2.imshow(selname, img)
    key = cv2.waitKey() & 0xFF
    print(key)
    if key==27:  #按ESC退出
        break
    elif key == 110:
        Recpoints = []
        ret, img = cap.read()

if len(Recpoints) < 4:
    print('must mark 4 point to continue')
    exit(0)

#转换后得到矩形的坐标点
realw = 153
realh = 273
scale = min(width / realw, height / realh)
realw = int(realw * scale)
realh = int(realh * scale)
#源图像中四边形坐标点（获取坐标点方法可参照我上篇博文）
point1 = np.array(Recpoints[:4], dtype = "float32")
#转换后得到矩形的坐标点
point2 = np.array([[0,0],[realw,0],[0,realh],[realw,realh]],dtype = "float32")
print('point1', point2)
print('point2', point2)
M = cv2.getPerspectiveTransform(point1,point2)
print('M', M)

def getMapPos(point):
    # 坐标映射矩阵
    pt = np.array((point[0], point[1], 1))
    pt2 = np.dot(M, pt)
    pt2 /= pt2[2]
    return pt2[0], pt2[1]

print("now check getMapPos with point1")
for pt in point1:
    print(pt, '=', getMapPos(pt))

while 1:
    ret, img = cap.read()
    if not ret:
        break
    cv2.imshow('img', img)
    out_img = cv2.warpPerspective(img,M,(realw,realh))
    cv2.imshow('trans', out_img)
    k = cv2.waitKey(30)
    if k == ord('q') or k == 27:
        break

cv2.destroyAllWindows()