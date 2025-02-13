import numpy as np  # type: ignore
import cvxpy as cp  # type: ignore
import socket as st



class MPC_timestep:
    
    
#不要init了，直接把矩阵全部写在class内
# 写在矩阵内的是全局变量，即便在外部模块调用（import）会伴随着调用模块的生命周期一直存在
# 也可以通过创建实例，利用初始化方法传入参数
# 问题点在于如何不断更新kf值？内部不断更新？还是在外部记录传参进来？










     def __init__(self, y_min, y_max, y_ref):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        
        # 权重矩阵
        self.Q_MPC = np.eye(2) * 0.4 # Q_MPC 状态权重
        self.R_MPC = np.eye(2) * 0.1 # R_MPC 控制输入权重
        self.F_MPC = F_MPC # 终端权重矩阵，惩罚终端状态

        self.u_min = np.array([[1],[0]])
        self.u_max = np.array([[5],[10]])
        self.y_min = y_min
        self.y_max = y_max
        self.y_ref = y_ref

        self.nx = 2 # A.shape[0]
        self.ny = 2 # C.shape[1]
        self.nu = 2 # B.shape[0]
        
        # 噪声协方差矩阵
        self.Q_KF = np.eye(2) * 0.1    #nx 过程噪声协方差
        self.R_KF = np.eye(2) * 0.1    #ny 测量噪声协方差
        self.P = np.eye(2)      # 误差协方差，维度应该与nx对应
        # self.K = K              # 卡尔曼增益矩阵 是不需要的，在后续直接通过P计算得到

        # 在每轮
        # x是浮点数，而u一定是整数，但是mpc给出的最优解不一定是整数，我们需要四舍五入
        # 
        self.x0 = np.array([[0.0],[0.0]])
        self.u0 = np.array([[1],[0]])
        

     

     #校正x 
     #传入上一时间步的x、u，y
     def KF(self, xp, up, y_observed):


        # KF
        
        # 预测
        #  x_old是估计值，x是校正值，x是覆盖更新的，t-1时刻的x也写作x，
        # 如果考虑可变I的话，那么上一时刻依然是t-1，只是控制向量u为0
        x_old = self.A @ xp + self.B @ up
        P_old = self.A @ self.P @ self.A.T + self.Q_KF

        # 校正
        self.K = P_old @ self.C.T @ np.linalg.inv(self.C @ P_old @ self.C.T + self.R_KF)
        x = x_old + self.K @ (y_observed - self.C @ x_old)
        self.P = (np.eye(self.nx) - self.K @ self.C) @ P_old

        return x 
    
     # 每次的horizon都是分别给定的
     # x0 就是本轮mpc的启示状态，等于x_prev

     # 没有添加噪声,不确定需不需要，因为真实数据中是存在噪声的，但是在测量过程中需要添加噪声
     def MPC(self, y_observed, H=5, I=0):


        # 创建一个矩阵符合维度/预测长度的矩阵

        x = cp.Variable((self.nx, H+1))  # 未来的状态
        u = cp.Variable((self.nu, H))    # 未来的输入
        y = cp.Variable((self.ny, H))
         # 初始化

        cost = 0

        # kf更新x
        # 此处x0,u0都是上一轮的x和u
        # x_est是下一轮的x的估计，但是因为这个是黑箱模型，无法观测到x的值，所以下一轮依然延用这个x
        # 因此可以提前更新x0
        x_est = self.KF(self.x0, self.u0, y_observed)
        self.x0 = x_est

        # flatten只是作为一个保障，保障x是一个行向量
        constraints = [x[:, 0] == x_est.flatten()]


        # 状态转移和/控制输入约束
        for i in range(H):
            constraints += [x[:, i+1] == self.A @ x[:, i] + self.B @ u[:, i]]
            constraints += [y[:, i] == self.C @ x[:, i]]                    #+ D @ u[:, i]
            constraints += [self.u_min <= u[:, i], u[:, i] <= self.u_max]   # 控制输入限制
            constraints += [self.y_min <= y[:, i], y[:, i] <= self.y_max]   # 状态限制
            # 假如y是一个性能指标（比如延迟、那么它就是一个尽可能小的值，而且有一个衰减因子）
            cost += cp.quad_form(y[:, i] - self.y_ref.flatten(), self.Q_mpc)\
                  + cp.quad_form(u[:, i], self.R_mpc)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        if problem.status == cp.OPTIMAL:
            u_optimal = u[:, 0].value  # 当前时刻的最优控制输入
            self.u0 = u_optimal
        else:
            pass    #需要向RL组件/main报错，按理说，应该延用上次计算的此时刻u作为备选，
                    #因此是否应该返回全部的u呢
         
        # 可能不是整数 需要取整
        return u_optimal







