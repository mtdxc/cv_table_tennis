# Tsai两步标定法

Tsai说：在对摄像机进行标定时如果考虑过多的非线性畸变会引入过多的非线性参数,这样往往不仅不能提高标定精度,反而会引起解的不稳定，与线性模型标定类似，只是正交了，理想成像平面到实际成像平面之间的转换。因此，Tsai两步标定法只考虑了径向畸变，也是大多数情况下会存在的畸变类型。  
**Tsai存在的弊端：** 无法通过一个平面标定全部的外参数，涉及非线性运算可能使得结果不稳定。  
**两步**：  
第一步利用最小二乘法解超定线性方程组,给出外部参数，求得的参数为：r1-r9，sx，tx，ty  
第二步求解内部参数,如果摄像机无透镜畸变,可一个超定线性方程解出,如果存在径向畸变,则通过一个三变量的优化搜索求解，求得参数为：有效焦距f, T中的tz和透镜畸变系数k  
**存在径向畸变的坐标系之间的关系：**  
![](zbx.png)  
Pu：理想成像点，pd：实际成像点，畸变没有导致方向发生改变  
$O_1p_d // O_1p_u // P_{oz} P$
由像素坐标系和世界坐标系之间的关系，添加畸变系数后得到如下公式，其中，$(x_c,y_c,z_c)$为某物在摄像机坐标系下的坐标，$（x_w,y_w,z_w）$为某物在世界坐标系下的坐标。将前面所述的矩阵关系写成方程  
的形式：
$ 
x_c = r_1 x_w + r_2 y_w + r_3 z_w + t_x \\
y_c = r_4 x_w + r_5 y_w + r_6 z_w + t_y \\
z_c = r_7 x_w + r_8 y_w + r_9 z_w + t_z \\
$

由于方向一致，及方向平行：  
RAC意味着$\frac{x_d}{y_d} = \frac{x_c}{y_c} = \frac{r_1 x_w + r_2 y_w + r_3 z_w + t_x}{r_4 x_w + r_5 y_w + r_6 z_w + t_y}$
整理得：  
$
\begin{bmatrix} x_w y_d & y_w y_d & z_w y_d & y_d & - x_w x_d & - y_w x_d & - z_w x_d \end{bmatrix}
\begin{bmatrix} r_1 / t_y \\ r_2/t_y \\ r_3/t_y \\ t_x/t_y \\ r_4/t_y \\ r_5/t_y \\ r_6/t_y \end{bmatrix} = x_d
$
将标定板设置为Z平面，可选取世界坐标系z=0（则含有$z_w$项为零）  
$
\begin{bmatrix} x_w y_d & y_w y_d & y_d & - x_w x_d & - y_w x_d \end{bmatrix}
\begin{bmatrix} r_1 / t_y \\ r_2/t_y \\ t_x/t_y \\ r_4/t_y \\ r_5/t_y \end{bmatrix} = x_d
$
对于一张图像中的N个点进行计算，上式可以修改如下：
$
r_1^’ = r_1 / t_y, r_2^’ = r_2 / t_y, t_x^’ = t_x / t_y, r_4^’ = r_4 / t_y, r_5^’ = r_5 / t_y \\
A = \begin{bmatrix}
x_{w1} y_{d1} & y_{w1} y_{d1} & y_{d1} & -x_{w1} x_{d1} & -y_{w1} x_{d1} \\
x_{w2} y_{d2} & y_{w2} y_{d2} & y_{d2} & -x_{w2} x_{d2} & -y_{w2} x_{d2} \\
... & ... \\
x_{wN} y_{dN} & y_{wN} y_{dN} & y_{dN} & -x_{wN} x_{dN} & -y_{wN} x_{dN} \\
\end{bmatrix} \\
X = \begin{bmatrix} r_1^’ & r_2^’ & t_x^’ & r_4^’ & r_5^’\end{bmatrix}^T \\
Y = \begin{bmatrix} x_{d1} & x_{d2} & x_{d1} & x_{d3} & ... & x_{dN} \end{bmatrix}^T \\
则 X 的最小二乘估计为: x^’ = (A^T A)^{-1} A^T Y
$
此时，求得$r_1，r_2，t_x，r_4，r_5$  
利用R（旋转矩阵）的正交性，求得$t_y，r_1—r_9$  
此时求得了相机模型的**外部参数**  

# 第二步：求解内部参数  
设置畸变系数K=0为初始值，暂时不考虑K得到超定方程组。  
$
[y_i - d_y(x_{fi} - u_0)] \begin{bmatrix} f \\ t_z \end{bmatrix} = w_i d_y (y_{fi} - v_0) \\
y_i = r_4 x_{wi} + r_5 y_{wi} + r_6 * 0 + t_y \\
w_i = r_7 x_{wi} + r_8 y_{wi} + r_9 * 0 
$
求得f和tz，作为初始值，使用优化算法进行迭代更新，得到更精确的相机参数$k，f，t_z$，比如最小二乘法。