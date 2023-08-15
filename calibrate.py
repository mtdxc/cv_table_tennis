import numpy as np
import cv2

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 结论：
# 棋盘格必须最少3x3, 但此时返回的dist误差较大
# 棋盘数只需要1个就能calibrateCamera
# objp坐标时要保证Z为0，
# 棋盘格检测objp是按X先变化，后Y变化，从小到大顺序写，
# 手工模拟填充点时不需要这样，只需保证Z为0就行
grid = (9, 6)
# grid = (3, 3)
imgname = 'calibrate with %dx%d' % grid
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((grid[0] * grid[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1,2)
objp *= 28.7 #MM
print(objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

calibrate = False
global ret, mtx, dist, rvecs, tvecs

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
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def undistort1(frame):
    h,w=frame.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    # undistort
    return cv2.undistort(frame, mtx, dist, None, newcameramtx)

# 帧采集个数
frameNo = 0
# 上次匹配的棋盘帧号，避免匹配后连续使用多帧图像
lastChessNo = 0

# 打开摄像头
cap = cv2.VideoCapture(0)
while(cap.isOpened()): #循环读取每一帧0
    ret, img = cap.read()
    if not(ret):
        break
    # 增加帧数
    frameNo = frameNo + 1
    h,  w = img.shape[:2]
    if calibrate:
        # undistort
        dst = undistort(img)

        # show image
        cv2.putText(dst, 'frame %d, undistort'%frameNo, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow(imgname, dst)
    else:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, grid, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, grid, corners, ret)
            # 检测到棋盘后，应该略过多少帧，再进行检测
            if (frameNo - lastChessNo) > 20:
                # 亚像素级优化
                corners2 = cv2.cornerSubPix(gray,corners,(grid[0],grid[0]),(-1,-1),criteria)
                # 增加点
                imgpoints.append(corners2)
                objpoints.append(objp)
                # 更新帧号
                lastChessNo = frameNo

            # 检验N个棋盘后进行相机校准，可修改这个值进行
            if len(objpoints) > 1:
                print('objpoints', objpoints)
                print('imgpoints', imgpoints)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
                if ret:
                    calibrate = ret
                    print('mtx', mtx)
                    print('dist畸变值', dist)
                    print('rvecs 旋转（向量）外参', rvecs)
                    print('tvecs 平移（向量）外参', tvecs)
                    print('==============================')
                else:
                    print('calibrateCamera error')

        cv2.putText(img, 'frame %d, %d chessboard detected'%(frameNo, len(objpoints)), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow(imgname, img)

    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('r'):
        # 重新校准
        print('reset data and redo calibrate')
        calibrate = False
        objpoints = []
        imgpoints = []

# 释放资源
cap.release()
cv2.destroyAllWindows()

