
- 世界坐标 -> 相机坐标
$$
\begin{bmatrix} x \\ y \\ z \end{bmatrix} = 
R \begin{bmatrix} x_w \\ y_w \\ z_w \end{bmatrix} + T 
$$

- 相机坐标 -> 图像坐标
$$
x_u = f \frac{x}{z}, y_u = f \frac{y}{z}
$$

引入畸变量 $x_d + D_x = x_u, y_d + D_y = y_u$ 取一阶径向畸变量
结合径向约束条件和畸变，tsai标定法的数学模型为：
$$
f \frac{r_1 x_w + r_2 y_w + r_3 z_w + t_x}{ r_7 x_w + r_8 y_w + r_9 z_w + t_z} = x_d (1 + k r^2)\\
f \frac{r_4 x_w + r_5 y_w + r_6 z_w + t_y}{ r_7 x_w + r_8 y_w + r_9 z_w + t_z} = y_d (1 + k r^2)\\
$$

## 第一步，由径向约束条件
有
$$
\frac{x_c}{y_c} = \frac{x_d}{y_d} = \frac{r_1 x_w + r_2 y_w + r_3 z_w + t_x}{r_4 x_w + r_5 y_w + r_6 z_w + t_y}
$$
得到
$$
x_w y_d \frac{r_1}{t_y} + y_w y_d \frac{r_2}{t_y} + z_w y_d \frac{r_3}{t_y} + y_d \frac{t_x}{t_y} - x_w x_d \frac{r_4}{t_y} - y_w x_d \frac{r_5}{t_y} - z_w x_d \frac{r_6}{t_y} = x_d
$$
简化为$A B = x_d$，
其中 $A = \begin{bmatrix} x_w y_d & y_w y_d & z_w y_d & y_d & - x_w x_d & - y_w x_d & - z_w x_d \end{bmatrix}$ 已知，
$B = \begin{bmatrix} r_1 / t_y & r_2/t_y & r_3/t_y & t_x/t_y & r_4/t_y & r_5/t_y & r_6/t_y \end{bmatrix} ^T$ 待求

已知 7 个 以上标注点的世界坐标和像平面坐标，可求得 7 个分量

----
## 改进
为进一步简化运行，选取共面的标注板上的点作为标注点，选择世界坐标系的Z原点为标注版，即$z_w=0$,
得到 $A^’ B^’ = x_d$
其中 $A^’ = \begin{bmatrix} x_w y_d & y_w y_d & y_d & - x_w x_d & - y_w x_d \end{bmatrix}$
$B^’ = \begin{bmatrix} r_1 / t_y & r_2/t_y & t_x/t_y & r_4/t_y & r_5/t_y \end{bmatrix} ^T$
需要特征点数 N >=5， House holder变化求解线性最小二乘解，得出5个分量
$ r_1^’ = r_1 / t_y, r_2^’ = r_2 / t_y, r_3^’ = t_x / t_y, r_4^’ = r_4 / t_y, r_5^’ = r_5 / t_y $

故旋转矩阵 $R = 
\begin{bmatrix} 
 r_1^’ t_y & r_2^’ t_y & r_3\\
 r_4^’ t_y & r_5^’ t_y & r_6 \\
 r_7 & r_8 & r_9 \\
\end{bmatrix}$  
由R的正交性可得$ |r_i|=1, |r_j| = 1 $, $r_i$为R的列向量，$r_j$为R的行向量，得出
$$
\begin{bmatrix} 
 r_1^’ t_y & r_2^’ t_y & +- \sqrt{1-t_y^2(r_1^{’2} + r_2^{’2})} \\
 r_4^’ t_y & r_5^’ t_y & +- \sqrt{1-t_y^2(r_4^{’2} + r_5^{’2})} \\
 +- \sqrt{1-t_y^2(r_1^{’2} + r_4^{’2})} & +- \sqrt{1-t_y^2(r_2^{’2} + r_5^{’2})} & +- \sqrt{s_r t_y^2 - 1} \\
\end{bmatrix} \\
其中 s_r = r_1^{’2} + r_2^{’2} + r_4^{’2} + r_5^{’2} \\
$$
利用 R 的正交性 $ r_i r_j = 0 $ => 
$$ (r_1^’ r_5^’ - r_2^’ r_4^’)^2 t_y^4 - s_r t_y ^ 2 + 1 = 0 $$
根据
$$
a x^2 + b x + c = 0 \\
x = \frac{-b +- \sqrt{b^2 - 4 a c}}{4a}
$$
解得 
$$
t_y^2 = \frac{s_r - \sqrt{s_r^2 - 4 (r_1^’ r_5^’ - r_2^’ r_4^’)^2}}{2 (r_1^’ r_5^’ - r_2^’ r_4^’)^2}
$$ 
=> $
|t_y|的值, x_c x_d 同号，y_c y_d同号，假设 t_y为正, \\
得到 r_1 = r_1^’ t_y, r_2 = r_2^’ t_y, t_x = r_3^’ t_y, r_4 = r_4^’ t_y, r_5 = r_5^’ t_y \\
任选一点 P_n(x_{wn}, y_{wn}, 0), 其摄像机坐标为 (x_{cn}, y_{cn}, z_{cn}), 像素坐标为 (x_{dn}, y_{dn})，有 \\
x_{cn} = r_1 x_{wn} + r_2 y_{wn} + t_x \\
y_{cn} = r_4 x_{wn} + r_5 y_{wn} + t_y \\
若 x_{cn}, x_{dn} 同号, 且 y_{cn}, y_{dn} 同号，则 t_y 为正，反之为负
$
求得$t_x, r_1, r_2, r_4, r_5$，得到R的两种情况
R = $
\begin{bmatrix}
r1 & r2 & \sqrt{1-r_1^2 - r_2^2} \\
r4 & r5 & -sgn \sqrt{1-r_4^2 - r_5^2} \\
r7 & r8 & r9 \\
\end{bmatrix} 
 或 
\begin{bmatrix}
r1 & r2 & - \sqrt{1-r_1^2 - r_2^2} \\
r4 & r5 & sgn \sqrt{1-r_4^2 - r_5^2} \\
-r7 & -r8 & r9 \\
\end{bmatrix} \\
sgn表示 r_1 r_4 + r_2 r_5 的符号:\\
由正交化： r_1 r_4 + r_2 r_5 + r3 r6 = 0 => r3 r6 = -(r_1 r_4 + r_2 r_5) \\
$
$r_7,r_8,r_9$ 由前两行参数的叉乘得到:
$$
\begin{bmatrix}
r_7 \\ r_8 \\ r_9
\end{bmatrix} = \begin{bmatrix}
r_2 r_6 - r_3 r_5 \\ r_3 r_4 - r_1 r_6 \\ r_1 r_5 - r_2 r_4
\end{bmatrix} 
$$
具体选择哪一个R，选取任意一个进行计算，如果算出 f < 0, 则丢弃；若 f >=0，则 R 为正确值
