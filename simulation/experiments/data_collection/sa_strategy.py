from interfaces import AbstractSelfAdaptingStrategy
import numpy as np  
import cvxpy as cp 
import socket as st


class DataCollectionSelfAdaptingStrategy(AbstractSelfAdaptingStrategy):

 def __init__(self):
    pass
 

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
            if worker.get_attribute("temp"):
                active_worker_tmp.append(worker)
                active_tmp += 1
            else:
                active_worker_reg.append(worker)
                active_reg += 1

        else:
            if worker.get_attribute("temp"):
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

    reg = np.random.randint(1, 11)
    tmp = np.random.randint(0, 21)
    u = np.array([[reg], [tmp]])



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
    
   

    return u

