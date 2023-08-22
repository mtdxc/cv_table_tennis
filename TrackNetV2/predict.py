import os
import cv2
import argparse
import numpy as np

import torch

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str, default='../../1_00_04.mp4')
parser.add_argument('--model_file', type=str, default='models/model_best.pt')
parser.add_argument('--num_frame', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--save_dir', type=str, default='pred_result')
args = parser.parse_args()

video_file = args.video_file
model_file = args.model_file
num_frame = args.num_frame
batch_size = args.batch_size
save_dir = args.save_dir

file_name = video_file.split('/')[-1]
video_name = file_name[:-4]
video_format = file_name[-3:]
out_video_file = f'{save_dir}/{video_name}_pred.{video_format}'
out_csv_file = f'{save_dir}/{video_name}_ball.csv'

checkpoint = torch.load(model_file)
param_dict = checkpoint['param_dict']
model_name = param_dict['model_name']
num_frame = param_dict['num_frame']
input_type = param_dict['input_type']
print(param_dict)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
 
# Load model
model = get_model(model_name, num_frame, input_type).cuda()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Video output configuration
if video_format == 'avi':
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
elif video_format == 'mp4':
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
else:
    raise ValueError('Invalid video format.')

# Write csv file head
f = open(out_csv_file, 'w')
f.write('Frame,Visibility,X,Y\n')

# Cap configuration
cap = cv2.VideoCapture(video_file)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# 1 frame represent (1/fps) second IRL then we can calculate the corresponding timestamp t at each step with:
time_step=(1/fps)
# 将此值设为0，可开启逐帧预览
sleep_ms = int(time_step * 1000) 
out = None # cv2.VideoWriter(out_video_file, fourcc, fps, (w, h))

success = True
frame_count = 0
num_final_frame = 0
ratio = h / HEIGHT

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
# 定标参数
realw = 1000
realh = 1000
# calibrateCamera 必须有8个点，15个行列式
point3d = [[0,0,0], [realw,0,0], [0,300,0], [realw/2,300,0], [realw,300,0], [0,realh,0], [0,realh,88], [realw/2, realh, 0], [realw,realh,0]]
# 硬编码7个对应点
point2d = [[447, 161], [945, 156], [408, 200], [699, 188], [985, 195], [258, 353], [250, 284], [703, 349], [1123, 345]]
pos2 = 1
pos3 = 5

def showPoints(frame, point2d, point3d):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0)
    for i in range(len(point2d)):
        # 标出标定点索引和对应的3d坐标
        cv2.circle(frame, point2d[i], 4, (0, 255, 0), -1)
        cv2.putText(frame, '(%d,%d,%d)'%tuple(point3d[i]), (point2d[i][0] + 5, point2d[i][1]), font, 0.7, color, 2)

# 打印标定点        
success, frame = cap.read()
if success:
    showPoints(frame, point2d, point3d)
    cv2.imshow('img', frame)
    cv2.waitKey()

# calc projection投射 matrix 
def calibration(points2D, points3D):
    # constructing DLT matrix 
    M=[]
    n=len(points2D)

    for i in range(n):
        X,Y,Z = points3D[i,0], points3D[i,1], points3D[i,2]
        x,y = points2D[i,0], points2D[i,1]

        l1 = [X,Y,Z,1,  0,0,0,0,  -x*X,-x*Y,-x*Z,-x]
        l2 = [0,0,0,0,  X,Y,Z,1,  -y*X,-y*Y,-y*Z,-y]
        M.append(l1)
        M.append(l2)

    M = np.array(M)

    # Find the parameters (p34 can be fixed to 1 to reduce the number of equations needed to 11):
    # using singular value decomposition
    U, S, V = np.linalg.svd(M)

    # The parameters are in the last line of Vh and normalize them
    LL = V[-1, :] / V[-1, -1]
    
    # Camera projection matrix
    P = LL.reshape(3, 4)
    return P

# points2D = cv2.undistortPoints(np.array(point2d, dtype = "float32"), mtx, dist, P=nmtx).reshape(-1,2)
# 3x4 array 必须要求一个非面的点
# P = calibration(points2D, np.array(point3d, dtype = "float32"))
# print('P', P)

g =-981  # !! opposing gravitationnal acceleration must be in cm/s^2 !!
# 根据3个以上轨迹和时间戳，来获取第一点的3D运动参数(X0,vx, Y0,vy, Z0, vz)
def get_3D_traj(ball_frame_positions, timestamps, P):
    #print('pos', ball_frame_positions)
    #print('tsp', timestamps)
    ball_points2D=np.array(ball_frame_positions)

    A=[]
    B=[]
    n = len(ball_points2D)
    # P is 3x4 array
    # P_coeffs = P.reshape(12)
    
    # Rewriting the projection relational system x=PX in such a way that isolate the 6 traj parameters in a X vector such that
    # AX=B with A and B already known from P coefficients and input points/timestamps.
    # The loop below "constructs" the A and B matrixes of this system.
    for i in range(n):
        t = timestamps[i]
        x, y = ball_points2D[i,0], ball_points2D[i,1]
        # print(x,y)

        p11,p12,p13,p14 = P[0]
        p21,p22,p23,p24 = P[1]
        p31,p32,p33,p34 = P[2]

        l1=[p11-x*p31,  p11*t-x*p31*t,    p12-x*p32,    p12*t-x*p32*t,   p13-x*p33,   p13*t-x*p33*t]
        l2=[p21-y*p31,  p21*t-y*p31*t,    p22-y*p32,    p22*t-y*p32*t,   p23-y*p33,   p23*t-y*p33*t]

        A.append(l1)
        A.append(l2)

        r1=[x*((1/2)*g*(t**2)*p33 + p34) - ((1/2)*g*(t**2)*p13 + p14)]
        r2=[y*((1/2)*g*(t**2)*p33 + p34) - ((1/2)*g*(t**2)*p23 + p24)]

        B.append(r1)
        B.append(r2)

    A=np.array(A)
    B=np.array(B)

    # optimally solves the system Ax=B with least squares method (minimize norm(Ax-B)) and find an approximate solution to 
    # this overdetermined system.
    opt_solved = np.linalg.lstsq(A,B,rcond=None)
    sol, residual, rank = opt_solved[0], opt_solved[1], opt_solved[2]
    if len(residual):
        print("Mean residual squared error : ", residual[0]/(2*n))

    X0,vx,Y0,vy,Z0,vz = sol.reshape(6)
    print('pos(%.2f,%.2f,%.2f) speed(%.2f,%.2f,%.2f)'%(X0, Y0, Z0, vx, vy, vz))
    return [X0,vx,Y0,vy,Z0,vz]


# Once we have the traj parameters, it is really simple to recover the real 3D points associated to the input sequence: 
# we just have to use the movement equations again.
def get_3D_points(traj_parameters, timestamps):
    
    [X0,vx,Y0,vy,Z0,vz] = traj_parameters
    ball_points3D = []
    n = len(timestamps)

    for i in range(n):
        t = timestamps[i]
        # 假设x,y都是匀速运动，而Z是自由落体运动
        ball_real_point = [X0+vx*t,  Y0+vy*t,  Z0+vz*t+(1/2)*g*(t**2)]
        ball_points3D.append(ball_real_point)

    return ball_points3D

# 获取球运动参数：起始点的3d坐标和初速度
# traj_parameters = get_3D_traj(ball_frame_positions, timestamps, P)
# 根据运动参数和时间戳，构造所有点的三维坐标
# ball_points3D = get_3D_points(traj_parameters, timestamps)
# print('3D ball real points sequence [X,Y,Z] associated to the input sequence of image points : ')
# print(ball_points3D)

# 抛物线检测
class PolDetect:
    def __init__(self, windowSize=4, polDiff=0.15):
        # 是否打印polWin日志
        self.debugPoint = True
        # 抛物线拟合阈值，当误差超过这值，就当其他抛物线
        self.polDiff = polDiff
        # 窗口大小，可用于过滤噪声点
        self.polWinSize = windowSize
        # 抛物线点窗口[[x,y,tsp]...], tsp为绝对时间戳
        self.polwin = []
        # 抛物线拟合参数
        self.coeffs = []

        # 抛物线上的所有点[[x,y,tsp]...], tsp为相对时间戳, 即polpts[0][2] = 0
        self.polpts = []
        # 第一个抛物线点的时间戳
        self.base_tsp = 0.0
        # 运动轨迹参数, 由get3DTraj更新
        self.traj = None

        # 摄像头内参
        self.mtx = None
        self.nmtx = None
        # 畸变参数
        self.dist = None
        # remap映射参数
        self.mapx = None
        self.mapy = None
        # 映射矩阵 = 内参 * 外参，也可通过calibration来调准
        self.P = None

        # 透视变化, 制造鸟瞰图
        # 将一个经投射后的四边形，映射成正规矩形
        self.M = None
    
    ####################################################
    # 摄像头校准
    def calibrate(self, imgSize, point2d, point3d):
        if len(point2d) != len(point3d) or len(point2d) < 5:
            print('args error')
            return False
        objp = []
        imgp = []
        # 过滤出共面点 Z=0
        for i in range(len(point3d)):
            if 0 == point3d[i][2]:
                objp.append(point3d[i])
                imgp.append(point2d[i])
        # 要求有共面的8点，即point3d的z相同
        objpoints = [np.array(objp, dtype=np.float32)]
        imgpoints = [np.array(imgp, dtype=np.float32)]

        # 校准参数    
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 只算k1畸变，算多的话，好像误差更大
        flag = cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K2 #| cv2.CALIB_FIX_TANGENT_DIST
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgSize, None, None, None, None, flag, criteria)
        if ret:
            print('calibrateCamera return', ret)
            print('int martix:', mtx)
            print('distort vec:', dist)
            print('rvecs', rvecs)
            print('tvecs', tvecs)
            self.mtx = mtx
            self.dist = dist

            # 必须设为0，否则会出黑边，且对不上
            nmtx,roi=cv2.getOptimalNewCameraMatrix(mtx,dist,imgSize,0)
            # 去畸变方法1
            # cv2.undistort(img, mtx, dist, None, nmtx)
            print('nmtx', nmtx)
            self.nmtx = nmtx

            self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx,dist,None,nmtx,imgSize,cv2.CV_32FC1)
            # 去畸变方法2，此方法效率会高点
            # cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

            # 还原出3x3旋转矩阵
            rvec,_ = cv2.Rodrigues(rvecs[0])
            # 3x4外参[R,T]
            ext = np.insert(rvec, 3, tvecs[0].reshape(-1), axis=1)
            print('ext matrix:', ext)
            # 算出映射矩阵：P = 内参*外参，
            # 这边只有一个面与calibration算出来的可能不一样，
            P1 = np.dot(mtx, ext)
            P1 = P1 / P1[-1,-1]
            print('P1:', P1)

            # 采用>6个不共面点，用calibration再计算一遍
            c2d = np.array(point2d, dtype=np.float32)
            c3d = np.array(point3d, dtype=np.float32)
            c2d = self.undistortPoints(c2d)
            P2 = calibration(c2d, c3d)
            print('P2:', P2)
            self.P = P1

            # 计算误差
            mean_error = 0
            total_error = 0
            for i in range(len(objpoints)):
                # 3d点转2d点
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2.reshape(-1,2), cv2.NORM_L2)
                mean_error += error/len(imgpoints2)
                total_error += error*error
            print("total error: {}".format(mean_error/len(objpoints)))
            print("rms error: {}".format(np.sqrt(total_error/(len(objpoints)*len(imgpoints2)))))

    def undistort(self, img):
        return cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)
    
    def undistortPoints(self, pts):
        pts = np.array(pts)
        # 去畸变, 要指定P=mtx，否则按归一化输出，值特别小
        return cv2.undistortPoints(pts, self.mtx, self.dist, P=self.nmtx).reshape(-1,2)
    
    #########################################################
    # 鸟瞰图
    def initPerspectiveTransform(self, ptRect, realRect):
        print('rect', ptRect)
        # 先去畸变
        uRect = self.undistortPoints(ptRect)
        print('undistort', uRect)
        print('real rect', realRect)
        # 然后再进行透视变化
        self.M = cv2.getPerspectiveTransform(uRect, realRect)
        print('Perspective martix', self.M)

    def getMapPos(self, point):
        if self.M is None:
            print('please call initPerspectiveTransform first')
        # 去畸变
        pt = self.undistortPoints(point)[0]
        # 坐标映射矩阵
        pt = np.array((pt[0], pt[1], 1))
        pt2 = np.dot(self.M, pt)
        pt2 /= pt2[2]
        return pt2[0], pt2[1]
    
    #########################################################
    # 3维重建
    def getTraj(self):
        return self.traj
    # 计算Traj参数，在确定抛物线(>3points)后计算的          
    def calc3DTraj(self):
        if self.P is None:
            print('please call calibrate first')
            return None
        n = len(self.polpts)
        if n < 3:
            print('pol points %d<3'%n)
            return None
        pts = np.array(self.polpts)
        #print('polpts', self.polpts)
        self.traj = get_3D_traj(pts[:,0:2], pts[:,2], self.P)
        return self.traj
    
    # 通过相对时间来获取3d位置
    def rTspTo3DPos(self, t):
        [X0,vx,Y0,vy,Z0,vz] = self.traj
        # 假设x,y都是匀速运动，而Z是自由落体运动
        return [X0+vx*t,  Y0+vy*t,  Z0+vz*t+(1/2)*g*(t**2)]
    
    # 返回绝对时间戳tsp的实时位置
    def get3DPos(self, tsp):
        if tsp < self.base_tsp:
            print('invalid tsp, must > %.2f', self.base_tsp)
            return [0,0,0]
        return self.rTspTo3DPos(tsp - self.base_tsp)

    # 返回整条抛物线的3d坐标
    def get3DPolPoints(self):
        ret = []
        if self.traj is None:
            print('please call get3DTraj fist')
        else:
            for tsp in np.array(self.polpts)[:2]:
                ret.append(self.rTspTo3DPos(tsp))
        return ret
        
    # 获取抛物线与Z平面的交点
    def getZeroZPts(self):
         ret = []
         if self.traj is None:
             return ret
         [X0,vx,Y0,vy,Z0,vz] = self.traj
         # 已知 Z0 + vz*t + (1/2)*g*(t**2) = 0
         # 求 [X0+vx*t,  Y0+vy*t]
         t1 = (-vz + np.sqrt(vz * vz - 2 * Z0 * g)) / (2 * Z0)
         ret.append([X0 - vx * t1, Y0 + vy * t1])
         t2 = (-vz - np.sqrt(vz * vz - 2 * Z0 * g)) / (2 * Z0)
         ret.append([X0 - vx * t2, Y0 + vy * t2])
         return ret
    
    
    #############################################
    # 抛物线检测

    # 返回预估的抛物线参数
    def coeffstr(self):
        ret = ''
        for val in self.coeffs:
            ret = ret + '%.2f,' % val
        return ret
    
    def addPoint(self, val):
        #self.debugPoint and print('addPoint', val)
        self.polwin.append(val)
        if len(self.polwin) > self.polWinSize:
            x = self.polwin.pop(0)
            self.debugPoint and print('popPoint', x)
    
    # 抛物线检测及3d坐标计算
    def update(self, x, y, tsp):
        diff = 0
        pt = [x, y, tsp]
        if len(self.coeffs):
            y_est = self.coeffs[0] + self.coeffs[1] * x + self.coeffs[2] * x**2
            # 在误差范围内就返回3d位置
            diff = abs(y_est/y-1)
            if diff < self.polDiff:
                self.debugPoint and print('matchPoint', pt)
                if (len(self.polwin) > 2):
                    print('pol change', self.coeffs)
                    self.base_tsp = self.polwin[0][2]
                    self.polpts = self.polwin
                    # 将绝对时间戳变成相对时间戳：以0开始
                    for pts in self.polpts:
                        pts[2] = pts[2] - self.base_tsp
                    # 计算追踪参数
                    self.calc3DTraj()
                # 将该点追加到抛物线点列表中，可用于轨迹重放
                self.polpts.append([x,y,tsp - self.base_tsp])
                # 贪心算法：匹配后就清空窗口，只有连续大于三个不匹配的点，才能确定新轨迹
                self.polwin = []
                return self.traj, '%.2f,%.2f,%.2f'%tuple(self.get3DPos(tsp))
            
        self.debugPoint and print('addPoint', pt, 'with diff', diff)
        # 误差范围外或点数不够，则加入待选数组中
        self.addPoint(pt)
        # 点数够则拟合抛物线参数
        if len(self.polwin) > 2:
            ary = np.array(self.polwin)
            # 拟合二阶抛物线参数
            self.coeffs = np.polynomial.polynomial.polyfit(ary[:,0], ary[:,1], 2)
        return None,''
 
# 方向检测
class DirDetect:
    def __init__(self):
        self.lastx = 0
        self.lasty = 0
        self.xadd = False
        self.yadd = False

    def update(self, x, y, tsp):
        changed = False
        if x != self.lastx:
            add = x > self.lastx
            changed = add != self.xadd
            self.xadd = add
        if y != self.lasty:
            add = y > self.lasty
            changed = add != self.yadd
            self.yadd = add
        print('%.2fs (%d,%d)'%(tsp, x, y))
        self.lastx = x
        self.lasty = y
        if changed:
            return None,'changed'
        return None,''

detect = PolDetect()
# 进行摄像头校准
detect.calibrate((w,h), point2d, point3d)

#############################
# 生成和校验鸟瞰图
print('\n>>> PerspectiveTransform test')
ptRect1 = np.array([point2d[0][:2], point2d[pos2][:2], point2d[pos3][:2], point2d[-1][:2]], dtype = "float32")
ptRect2 = np.array([point3d[0][:2], point3d[pos2][:2], point3d[pos3][:2], point3d[-1][:2]], dtype = "float32")
detect.initPerspectiveTransform(ptRect1, ptRect2)
print("now check getMapPos with point1")
for pt in ptRect1:
    print(pt, '=', detect.getMapPos(pt))
print('=== PerspectiveTransform test\n')

while success:
    print(f'Number of sampled frames: {frame_count}')
    basetsp = frame_count * time_step
    # Sample frames to form input sequence
    frame_queue = []
    for _ in range(num_frame*batch_size):
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect.undistort(frame)
            frame_count += 1
            frame_queue.append(frame)
    
    if not frame_queue:
        break
    
    # If mini batch incomplete
    if len(frame_queue) % num_frame != 0:
        frame_queue = []
        # Record the length of remain frames
        num_final_frame = len(frame_queue)
        # Adjust the sample timestampe of cap
        frame_count = frame_count - num_frame*batch_size
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        # Re-sample mini batch
        for _ in range(num_frame*batch_size):
            success, frame = cap.read()
            if not success:
                break
            else:
                frame = detect.undistort(frame)
                frame_count += 1
                frame_queue.append(frame)
        assert len(frame_queue) % num_frame == 0
    
    x = get_frame_unit(frame_queue, num_frame)
    
    # Inference
    with torch.no_grad():
        y_pred = model(x.cuda())
    y_pred = y_pred.detach().cpu().numpy()
    h_pred = y_pred > 0.5
    h_pred = h_pred * 255.
    h_pred = h_pred.astype('uint8')
    h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)
    
    for i in range(h_pred.shape[0]):
        if num_final_frame > 0 and i < (num_frame*batch_size - num_final_frame):
            # Special case of last incomplete mini batch
            # Igore the frame which is already written to the output video
            continue 
        else:
            img = frame_queue[i].copy()
            # cv2.imshow('heatmap', h_pred[i])
            cx_pred, cy_pred = get_object_center(h_pred[i])
            cx_pred, cy_pred = int(ratio*cx_pred), int(ratio*cy_pred)
            vis = 1 if cx_pred > 0 and cy_pred > 0 else 0
            # Write prediction result
            f.write(f'{frame_count-(num_frame*batch_size)+i},{vis},{cx_pred},{cy_pred}\n')
            if cx_pred != 0 or cy_pred != 0:
                _,ret = detect.update(cx_pred, cy_pred, basetsp + i * time_step)
                if len(ret):
                    cv2.putText(img, ret, (cx_pred + 5, cy_pred), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
                cv2.circle(img, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
            cv2.imshow("img", img)
            key = cv2.waitKey(sleep_ms)
            if key == 27:
                success = False
            if out:
                out.write(img)
if out:
    out.release()
print('Done.')