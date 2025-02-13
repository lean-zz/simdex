from interfaces import AbstractSelfAdaptingStrategy
from mpc_1timestep import MPC_timestep
import numpy as np  
import cvxpy as cp 
import socket as st

class mpcSelfAdaptingStrategy(AbstractSelfAdaptingStrategy):

 def __init__(self):
    pass
 
# 需要与mpc模块对接，需要通过mpc模块给定worker的值，然后再根据情况调整，后续需要增减workerqueue列表，或者增加新的特性
# 需要两个值，tmp，reg，怎么保存
def do_adapt(self, ts, dispatcher, workers, job, pd, pmd):
    
    # 先确定目前队列状态
    active_tmp = 0
    active_reg = 0
    inactive_tmp = 0
    inactive_reg = 0
    
    active_worker_tmp = []
    active_worker_reg = []
    inactive_worker_tmp = []
    inactive_worker_reg = []
    
    for worker in workers:
        if worker.get_attribute("active"):
            if worker.get_attribute("tmp"):
                active_worker_tmp.append(worker)
                active_tmp += 1
            else:
                active_worker_reg.append(worker)
                active_reg += 1

        else:
            if worker.get_attribute("tmp"):
                inactive_worker_tmp.append(worker)
                inactive_tmp += 1
            else:
                inactive_worker_reg.append(worker)
                inactive_reg += 1
    
    active_worker_tmp.sort(key=lambda x: x.get_finish_ts())
    active_worker_reg.sort(key=lambda x: x.get_finish_ts())

    inactive_worker_tmp.sort(key=lambda x: x.get_finish_ts())
    inactive_worker_reg.sort(key=lambda x: x.get_finish_ts())

    # 得到tmp、reg的值
    # H, I没有输入
    # y是period_max_delay，period_delay
    y_observed = np.array([[pd],[pmd]])
    mpc = MPC_timestep(y_observed)
    u_opt = mpc.MPC(x0 = np.array([]), H=5, I=0)
    reg = u_opt[0]
    tmp = u_opt[1]
    





    # 根据给定的tmp、reg调整workerqueue


    if active_tmp-tmp > 0:
        for i in range(active_tmp-tmp):
            active_worker_tmp[i-1].set_attribute("active", False)
    elif active_tmp-tmp < 0:
        for i in range(tmp-active_tmp):
            inactive_worker_tmp[i-1].set_attribute("active", True)
    else:
        pass

    if active_reg-reg > 0:
        for i in range(active_reg-reg):
            active_worker_reg[i-1].set_attribute("active", False)
    elif active_reg-reg < 0:
        for i in range(reg-active_reg):
            inactive_worker_reg[i-1].set_attribute("active", True)
    else:
        pass
    
    


    







    
    
    
    
    
    
    
    
    
    pass









class MPC_timestep:
    
    


     def __init__(self, A, B, C, D, Q_MPC, R_MPC, F_MPC,
                 Q_KF, R_KF, P, K,
                 u_min, u_max, y_min, y_max, y_ref):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.Q_MPC = Q_MPC
        self.R_MPC = R_MPC
        self.F_MPC = F_MPC

        self.u_min = u_min
        self.u_max = u_max
        self.y_min = y_min
        self.y_max = y_max
        self.y_ref = y_ref

        self.nx = A.shape[0]
        self.ny = C.shape[1]
        self.nu = B.shape[0]

        self.Q_KF = Q_KF    #nx
        self.R_KF = R_KF    #ny
        self.P = P      #误差协方差，维度应该与nx对应
        self.K = K

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
     def MPC(self, H, I, x0):

        x = cp.Variable((self.nx, H+1))  # 未来的状态
        u = cp.Variable((self.nu, H))    # 未来的输入
        y = cp.Variable((self.ny, H))
         # 初始化
        constraints = [x[:, 0] == x0]

        cost = 0



        # 状态转移和/控制输入约束
        for i in range(self.H):
            constraints += [x[:, i+1] == self.A @ x[:, i] + self.B @ u[:, i]]
            constraints += [y[:, i] == self.C @ x[:, i]]                    #+ D @ u[:, i]
            constraints += [self.u_min <= u[:, i], u[:, i] <= self.u_max]   # 控制输入限制
            constraints += [self.y_min <= y[:, i], y[:, i] <= self.y_max]   # 状态限制
            # 
            cost += cp.quad_form(y[:, i] - self.y_ref.flatten(), self.Q_mpc)\
                  + cp.quad_form(u[:, i], self.R_mpc)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        if problem.status == cp.OPTIMAL:
            u_optimal = u[:, 0].value  # 当前时刻的最优控制输入
        else:
            pass    #需要向RL组件/main报错，按理说，应该延用上次计算的此时刻u作为备选，
                    #因此是否应该返回全部的u呢


        return u_optimal

 
