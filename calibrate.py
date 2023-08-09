import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 结论：
# 棋盘格必须最少3x3, 但此时返回的dist误差较大
# 棋盘数只需要1个就能calibrateCamera
# objp坐标时要保证Z为0，按照X先变化，后Y变化，从小到大顺序写
grid = (9, 6)
# grid = (3, 3)
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
cap = cv2.VideoCapture(0)#创建一个 VideoCapture 对象
frameNo = 0
lastChessNo = 0
while(cap.isOpened()): #循环读取每一帧0
    ret, img = cap.read()
    if not(ret):
        break
    frameNo = frameNo + 1
    h,  w = img.shape[:2]
    if calibrate:
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        cv2.putText(dst, 'frame %d, undistort'%frameNo, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('img', dst)
        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
    else:
        ret = False
        if (frameNo - lastChessNo) > 10:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, grid, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(grid[0],grid[0]),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, grid, corners2, ret)
            lastChessNo = frameNo

            if len(objpoints) > 0:
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
                if ret:
                    calibrate = ret
                    print('mtx', mtx)
                    print('dist', dist)
                    print('rvecs', rvecs)
                    print('tvecs', tvecs)
                    print('==============================')
        cv2.putText(img, 'frame %d, %d chessboard detected'%(frameNo, len(objpoints)), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('img', img)
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('r'):
        print('reset data and redo calibrate')
        calibrate = False
        objpoints = []
        imgpoints = []
cap.release()
cv2.destroyAllWindows()

