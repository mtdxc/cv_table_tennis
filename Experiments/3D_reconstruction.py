import cv2
import numpy as np
import json
import itertools
from numpy.polynomial import polynomial as pol

L=275
W=152.5
NH=11
print('%.1fx%.1f NH=%d'%(L, W, NH))
# 直接将物理坐标映射到乒乓球桌上，将世界坐标原点设置O点，即乒乓球桌一边的中点
# O=[0,0,0]
A=[W/2,0,0]
B=[-W/2,0,0]
C=[-W/2,L,0]
D=[W/2,L,0]
CENTER_DOWN=[0,L/2,0]
CENTER_UP=[0,L/2,NH]
M1_U=[W/2,L/2,NH]
M2_U=[-W/2,L/2,NH]
M1_L=[W/2,L/2,0]
M2_L=[-W/2,L/2,0]

# 在图片上找出关键点的二维坐标
# Retrieving the values of key points on the image so that the user can skip the above pts-selection part
a,b,m2_l,m2_u,c,d,m1_l,m1_u=([1267, 1030],
 [180, 966],
 [595, 790],
 [598, 727],
 [781, 709],
 [1267, 720],
 [1269, 819],
 [1267, 742]
)

dim_img=cv2.imread('real_dim_and_cd.jpg')
cv2.imshow('table', dim_img)
cv2.waitKey(0)

print('\n-----calibration-------')
# solving system to find camera proj matrix P using DLT : needs at least 6 points given P is 3*4

# pairs of corresponding points, a least 5 points
points2D=np.array([a,b,m2_l,m2_u,c,d,m1_l,m1_u])
points3D=np.array([A,B,M2_L,M2_U,C,D,M1_L,M1_U])
    
# 做透视变化
point1 = np.array([a, b, c, d],dtype = "float32")
point2 = np.array([A[:2], B[:2], C[:2], D[:2]], dtype = "float32")
print(point1, point2)
M = cv2.getPerspectiveTransform(point1, point2)
print('M', M)
def PointTo2d(pt):
    # 坐标映射矩阵
    pt = np.array((pt[0], pt[1], 1))
    pt2 = np.dot(M, pt)
    pt2 /= pt2[2]
    return pt2[0], pt2[1]

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

# 3x4 array
P = calibration(points2D, points3D)
print('P', P)

# maps 3D pts to 2D
def Map3DTo2d(x, y, z):
    pt3d = np.array([x, y, z, 1])
    pt2d = np.dot(P, pt3d.reshape((4,1)))
    # normal 2d
    pt2d = pt2d / pt2d[2]
    return int(pt2d[0]), int(pt2d[1])

# verif that P correctly maps 3D pts to 2D : verification some key table points
print("middle point in image is", Map3DTo2d(0, L/2, 0))
# 错误的, 只有在z为0时, 此次映射变化才生效
print("middle point in table is", PointTo2d(m1_u))
print("middle point in table is", PointTo2d(m1_l))
print('\n-----factorize-------')
import scipy
import scipy.linalg as linalg

# 计算相机的内参和外参
def factorize(P):
    # We know from the course that P=[M|m] = K[R|t], and that m verifies m=-MC, showing that Kt=-KRC and t=-RC
    M = P[:,0:3]
    
    # M=KR : apply QR-decomp to M, and get K,R by simple identification
    K,R = scipy.linalg.rq(M)
    
    # We want K to have a positive diagonal (positive focal lenghts), then if it is not the case we can "correct" the K,R
    # solution with : K'=K*T and R'=T*R, thus ensuring a >0 diag for K and that the K,R solution is still correct as:
    # K'R'=K(TT)R=KR=M
    T = np.diag(np.sign(np.diag(K)))
    K = np.dot(K,T)
    K = K/K[2,2]
    R = np.dot(T,R)
    
    # m=-M*c shows that C=-inv(M)*m = - inv(M)*last_col(P)
    C = np.dot(scipy.linalg.inv(-M),P[:,3])
    return (K,R,C)

    
K,R,C = factorize(P)

print('Intrinsic parameters matrix K')
print(K.round(2),'\n')

print('Extrinsic rotation matrix R')
print(R.round(2),'\n')

print('Camera center position in RW C')
print(C.round(2),'\n')


print('\n-----load data-------')
draw_trajectory = cv2.imread('seq.jpg')
# 从tracker.py中读取抛物线点
ball_frame_positions = []
timestamps = []
with open('allpt.json', 'r') as file_to_write:
    obj = json.load(file_to_write)
    ball_frame_positions = obj['positions']
    timestamps = obj['timestamps']
print('load positions', ball_frame_positions, ',timestamp', timestamps)

print('\n-----clean data-------')
# 清洗数据
# Function to get all the subsets of a vector (needed for RANSAC algo)
def sub_vectors(vector,k):
    n=len(vector)
    COMB=itertools.combinations(range(n), k)
    sub_vectors=[]
    
    for c in COMB:
        indexes=list(c)
        sub_vector=[vector[index] for index in indexes]
        sub_vectors.append(sub_vector)
        
    sub_vectors=np.array(sub_vectors)
    return sub_vectors


# RANSAC algo for parabola fitting 抛物线过滤 : 
# iteratively fitting a parabola through all subsets of 2D points, 
# and for each parabola fitted this way, see with how much points (cf C, consensus) it corresponds (e.g error < epsilon!). 
# The maximal consensus obtained through all the algo should correspond to the 'inliers' points 
# (some points mike look like inliers but will be rejected because too 'innacurately detected 不准确地检测到').
def RANSAC(timestamps, ball_frame_positions, k, epsilon):

    MAXC=0
    V = sub_vectors(ball_frame_positions, k)
    inliers=[]
    timestamp_inliers=[]
    
    for vector in V:
        
        X=vector[:,0]
        Y=vector[:,1]
        # 多项式拟合
        # 根据用户给定的x、y、多项式阶数等信息拟合多项式。
        # 返回值是一个系数向量。
        # 原理：最小二乘拟合，即使得平方误差最小化。(Y -y) ^ 2 之和最小
        # y = coeffs[0] + coeffs[1] * x + coeffs[2] * x^2
        coeffs = pol.polyfit(X,Y,2)
        
        consensus=[]
        t_consensus=[]
        
        # finding all the valid points in the input sequence, given the current parabola P and the tolerance param eps
        for i in range(len(ball_frame_positions)):
            [x, y] = ball_frame_positions[i]
            t = timestamps[i]
            y_est = coeffs[0] + coeffs[1] * x + coeffs[2] * x**2
            err = abs(y - y_est) / y
            if err < epsilon:
                consensus.append([x,y])
                t_consensus.append(t)
                
        # updating maximal consensus set
        C = len(consensus)
        if C > MAXC:
            MAXC=C
            inliers = consensus
            timestamp_inliers = t_consensus

    return inliers, timestamp_inliers
        
inliers,timestamp_inliers = RANSAC(timestamps, ball_frame_positions, 3, 0.01)
print('use data', inliers, timestamp_inliers)

ball_frame_positions = np.array(inliers)
timestamps = np.array(timestamp_inliers)

# 显示出可用的点
for i in range(len(ball_frame_positions)):
    ball_position=[ball_frame_positions[i,0], ball_frame_positions[i,1]]
    cv2.circle(draw_trajectory, ball_position, 7,(0, 255, 0), 2)

cv2.imshow('trajectory', draw_trajectory)
cv2.waitKey(0)

print('\n-----3D reconstruction-------')
# methods using physics to find the ball 6 real traj parameters (namely X0,vx, Y0,vy, Z0, vz), 
# knowing that each 2D point raises 2 equations ==> needs at least 3 points to raise 6 indep equations 
# ===> RANSAC should not discard too much points as we saw before!

# 保证第一个时间戳为0
# Beforeall, it is necessary to have timestamp. We fix origin of time t0=0s as the timestamp of the first sequence point.
t0 = timestamps[0]
timestamps=[(t-t0) for t in timestamps]

g =-981  # !! opposing gravitationnal acceleration must be in cm/s^2 !!
# 根据3个以上轨迹和时间戳，来获取第一点的3D运动参数(X0,vx, Y0,vy, Z0, vz)
def get_3D_traj(ball_frame_positions, timestamps, P):
    
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
    sol, residual, rank = opt_solved[0], opt_solved[1][0], opt_solved[2]

    print("Mean residual squared error : ", residual/(2*n))

    X0,vx,Y0,vy,Z0,vz = sol.reshape(6)
    print('X0 : ',X0,', vx : ',vx,', Y0 : ',Y0,', vy : ',vy,', Z0 : ',Z0,', vz : ',vz)
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
traj_parameters = get_3D_traj(ball_frame_positions, timestamps, P)
# 根据运动参数和时间戳，构造所有点的三维坐标
ball_points3D = get_3D_points(traj_parameters, timestamps)

print('3D ball real points sequence [X,Y,Z] associated to the input sequence of image points : ')
print(ball_points3D)

# 将模拟图显示在3d上..
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
def plot_in_3D(points3D,  # 球场标志点
               ball_points3D, # 球3d运动轨迹
               viewangle, # 视角
               traj_parameters):
    
    fig = plt.figure(figsize=(20,20))
    ax = fig.gca(projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(-150, 150)
    ax.set_ylim3d(0, 300)
    ax.set_zlim3d(0, 200)
    ax.view_init(elev=10., azim=viewangle)


    # Table points
    for i in range (len(points3D)):
    #     print(points3D[i])
        [X,Y,Z]=points3D[i]
        ax.scatter3D(X, Y, Z, c='yellow',alpha = 1,s =(100));

    ax.scatter3D(0, 0, 0, c='black',s =(100));   


    # plot table plan
    xx, yy = np.meshgrid(range(int(-W/2),int(W/2)),range(0,int(L)))
    eq = 0.0 * xx + 0.0 * yy
    ax.plot_surface(xx, yy, eq,alpha=0.1)

    # plot net plan
    xx, zz = np.meshgrid(range(int(-W/2),int(W/2)),range(0,int(NH)))
    yy=0*xx + (L/2)
    ax.plot_surface(xx, yy, zz,alpha=0.6,color='black')

    # plot ball points    
    for i in range (len(ball_points3D)):
    #     print(ball_points3D[i])
        [X,Y,Z]=ball_points3D[i]
        ax.scatter3D(X, Y, Z, c='red',s =(20),alpha=1)
        
    # opt : plot traj
    [X0,vx,Y0,vy,Z0,vz] = traj_parameters
    time= np.linspace(0, 0.5, 1000) # sec
    zline = Z0+vz*time+(1/2)*g*time**2
    xline = X0+vx*time
    yline = Y0+vy*time
    ax.plot3D(xline, yline, zline, 'gray',alpha=0.2)

# 显示会出错，暂时注释
# for angle in range(0, 360, int(360/5)):
#    plot_in_3D(points3D,ball_points3D,angle,traj_parameters)
#    plt.savefig('results/reconstructed_seq/reconstructed_seq'+str(nb)+'_'+str(angle)+'°'+'.jpg')