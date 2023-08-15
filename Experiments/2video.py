
# 功能：将图像中某一四边形变换为矩阵

import cv2
import numpy as np
import sys
choosen_video = '../../1_00_04.mp4'
if len(sys.argv) > 1:
    choosen_video = sys.argv[1]

cap = cv2.VideoCapture(choosen_video)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print('%s image size(%dx%d@%d)' % (choosen_video, width, height, fps))

#转换后得到矩形的坐标点
realw = 153
realh = 273
scale = min(width / realw, height / realh)
realw = int(realw * scale)
realh = int(realh * scale)
print('real size %dx%d' % (realw, realh))

nPoint = 0
# 3d坐标
point3d = []
# 初始化2d坐标为0，长度与point3d相同
point2d = []
# 当前选择点    
pos = 0
# 第二直角边位置
pos2 = 1
# 第三直角边位置
pos3 = 2
if nPoint > 0: # 平均生成3d点
    count = int(np.sqrt(nPoint))
    stepw = realw / (count - 1)
    steph = realh / (count - 1)
    pos2 = count - 1
    pos3 = count * pos2
    for i in range(count):
        hpos = steph * i
        for j in range(count):
            point3d.append([stepw*j, hpos, 0])
else: # 手工选择映射点
    #point3d = [[0,0,0],[realw,0,0],[0,realh,0],[realw,realh,0]]
    realw = 1000
    realh = 1000
    scale = 1
    # calibrateCamera 必须有8个点，15个行列式
    point3d = [[0,0,0], [realw,0,0], [0,300,0], [realw/2,300,0], [realw,300,0], [0,realh,0], [realw/2, realh, 0], [realw,realh,0]]
    # 编码7个对应点
    point2d = [[447, 160], [945, 156], [406, 200], [700, 188], [984, 195], [256, 361], [702, 356], [1130, 352]]
    pos = 7
    pos2 = 1
    pos3 = 5

nPoint = len(point3d)
print('use %d point3ds' % nPoint, point3d)

while len(point2d) < nPoint:
    point2d.append([0,0])

ret, img = cap.read()
if not ret:
    print('unable to open file ', choosen_video)
    exit()



def draw_circle(event,x,y,flags,param):
    global point2d
    global pos, nPoint
    # 双击后设置当前点，并移到下一个
    if event==cv2.EVENT_LBUTTONDBLCLK:
        point2d[pos] = [x,y]
        print(pos, '=', point2d[pos])
        pos = pos + 1

selname = 'select %d points of %dx%d' % (nPoint, realw, realh)
cv2.namedWindow(selname,cv2.WINDOW_NORMAL)
cv2.setMouseCallback(selname, draw_circle)

font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 0, 0)
# 手工选择标定点
while(pos < nPoint):
    frame = img.copy()
    for i in range(nPoint):
        # 标定点
        cv2.circle(frame, point2d[i], 4, (0, 255, 0), -1)
        cv2.putText(frame, "%d(%d,%d)" % (i, point3d[i][0], point3d[i][1]), 
                (point2d[i][0] + 5, point2d[i][1]), font, 0.7, color, 2)
    cv2.putText(frame, "select %d point(%d,%d)" % (pos, point3d[pos][0], point3d[pos][1]), 
            (int(width/2) - 50, 20), font, 0.7, color, 2)
    cv2.imshow(selname, frame)

    key = cv2.waitKey(20) & 0xFF
    if key==27:  #按ESC退出
        exit(0)
    elif key == ord('s'):
        break
    elif key >= ord('0') or key <= ord('9'):
        # 重设某个点
        newPos = key - ord('0')
        if newPos < nPoint:
            pos = newPos
            print('reset pos', pos)
    elif key == ord('n'):
        # 转到下一帧进行标定
        point2d = []
        for i in range(nPoint):
            point2d.append((0,0))
        pos = 0
        ret, img = cap.read()

calibrate = False
objpoints = [np.array(point3d, dtype=np.float32)]
imgpoints = [np.array(point2d, dtype=np.float32)]
flag = cv2.CALIB_FIX_K3 #| cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_TANGENT_DIST
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2], None,None, None, None, flag)
if ret:
    calibrate = ret
    print('martix', mtx)
    print('dist', dist)
    print('rvecs', rvecs)
    print('tvecs', tvecs)

def undistort(frame):
    h,w=frame.shape[:2]
    k=np.array(mtx)
    d=np.array(dist[0])
    mapx,mapy=cv2.initUndistortRectifyMap(k,d,None,k,(w,h),5)
    return cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)

def undistort2(frame):
    h,w=frame.shape[:2]
    # 如果缩放系数 alpha = 0,返回的非畸变图像会带有最少量的不想要的像素。它甚至有可能在图像角点去除一些像素。
    # 如果 alpha = 1,所有像素都会被返回,还有一些黑图像。它还会返回一个 ROI 图像,我们可用来对结果进行裁剪。
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    return dst

# 填充2d矩形坐标点
ptRect1 = np.array([point2d[0][:2], point2d[pos2][:2], point2d[pos3][:2], point2d[-1][:2]], dtype = "float32")
print('rect1', ptRect1)
# 去畸变, 暂时没用
print(cv2.undistortPoints(ptRect1, mtx, dist, ptRect1))
# 填充3d矩形
ptRect2 = np.array([point3d[0][:2], point3d[pos2][:2], point3d[pos3][:2], point3d[-1][:2]], dtype = "float32")
print('rect2', ptRect2)
M = cv2.getPerspectiveTransform(ptRect1,ptRect2)
print('M', M)

def getMapPos(point):
    # 坐标映射矩阵
    pt = np.array((point[0], point[1], 1))
    pt2 = np.dot(M, pt)
    pt2 /= pt2[2]
    return pt2[0], pt2[1]

print("now check getMapPos with point1")
for pt in ptRect1:
    print(pt, '=', getMapPos(pt))

while 1:
    ret, img = cap.read()
    if not ret:
        break
    if calibrate:
        img = undistort2(img)
    cv2.imshow('img', img)
    out_img = cv2.warpPerspective(img,M,(realw,realh))
    cv2.imshow('trans', out_img)
    k = cv2.waitKey(30)
    if k == ord('q') or k == 27:
        break

cv2.destroyAllWindows()