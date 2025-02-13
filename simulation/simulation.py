from workers import WorkerQueue
from interfaces import create_component
import numpy as np  
from sippy import system_identification as si

# simulation模块只有方法，主函数在main中，所以需要在最后将调用的传入main

def _create_instance(config, ref_jobs):
    """Helper function that creates instance of a component from configuration."""
    if isinstance(config, dict):
        # Basic type checks
        if ("class" not in config or "args" not in config):
            raise RuntimeError("Component configuration descriptor must have 'class' and 'args' properties.")

        # argument "@@ref_jobs" is replaced with ref_jobs list (special injection)
        if isinstance(config["args"], dict):
            args = {key: ref_jobs if val == "@@ref_jobs" else val for key, val in config["args"].items()}
        elif isinstance(config["args"], list):
            args = [ref_jobs if arg == "@@ref_jobs" else arg for arg in config["args"]]
        else:
            raise RuntimeError("Invalid component constructor args given in configuration descriptor.")

        return create_component(config["class"], args)
    else:
        return create_component(config)  # config is a string holding the class name


class Simulation:
    """Main simulation class. Wraps the algorithm and acts as component container."""

# 初始化里就需要初始化mpc组件（矩阵之类，通过yaml配置） 
# 或者直接作为mpc模块的全局变量
# simulation模块通过_create_instance方法，在main创建的simulation实例上已经添加了sa_stategy
# 通过self可以调用，既然是实例那么其变量就是实例变量，会一直伴随simulation实例整个生命周期存在

    def __init__(self, configuration, ref_jobs=None):
        # load parameters from configuration and instantiate necessary components
        self.metrics = []
        if "metrics" in configuration:
            for metric in configuration["metrics"]:
                self.metrics.append(_create_instance(metric, ref_jobs))

        self.dispatcher = _create_instance(configuration["dispatcher"], ref_jobs)
        if "sa_strategy" in configuration:
            self.sa_strategy = _create_instance(configuration["sa_strategy"], ref_jobs)
        else:
            self.sa_strategy = None  # strategy can be empty (i.e., no MAPE-K) for baseline ref. measurements

        # how often MAPE-K is called (in seconds)
        self.sa_period = float(configuration["period"]) if "period" in configuration else 60.0  # one minute is default

        # simulation state (worker queues)
        if "workers" not in configuration:
            raise RuntimeError("Workers are not specified in the configuration file.")

        self.workers = []
        if isinstance(configuration["workers"], list):
            for worker_attrs in configuration["workers"]:
                self.workers.append(WorkerQueue(**worker_attrs))
        else:
            for i in range(int(configuration["workers"])):
                self.workers.append(WorkerQueue())

        # remaining simulation variables
        self.ts = 0.0  # simulation time
        self.next_mapek_ts = 0.0  # when the next MAPE-K call is scheduled
        self.u_data_list = []
        self.y_data_list = []
        self.u = np.zeros((2,1))
        self.y = np.zeros((2,1))
        self.flag = False



    def register_metrics(self, *metrics):
        """Additional metrics components may be registered via this method (mainly for debugging purposes)."""
        for m in metrics:
            self.metrics.append(m)

    def __start_simulation(self, ts):
        """Just-in-time initialization."""
        self.ts = ts
        self.next_mapek_ts = ts + self.sa_period

        # 始化sa_strategy
        # initialize injected components
        self.dispatcher.init(self.ts, self.workers)
        if self.sa_strategy:
            self.sa_strategy.init(self.ts, self.dispatcher, self.workers)

        # take an initial snapshot by the metrics collectors
        for metric in self.metrics:
            metric.snapshot(self.ts, self.workers)

    def __advance_time_in_workers(self):
        for worker in self.workers:
            done = worker.advance_time(self.ts)
            for job in done:
                for metric in self.metrics:
                    metric.job_finished(job, self.ts)

    def __advance_time(self, ts):
        """Advance the simulation to given point in time, invoking MAPE-K periodically."""
        # 推进时间到目标ts
        job = None
        # 按照一个适应周期推进时间
        if self.metrics or self.sa_strategy:
            while self.next_mapek_ts < ts:
                self.ts = self.next_mapek_ts
                self.__advance_time_in_workers()

                # 生成metric
                pd = self.metrics[0].get_period_delay      # job delay
                pmd = self.metrics[0].get_period_max_delay
                pw = self.metrics[1]     # power


                # take a measurement for statistics
                for metric in self.metrics:
                    metric.snapshot(self.ts, self.workers)

                # invoke MAPE-K, the strategy can read and possibly update worker queues
                if self.sa_strategy:
                    self.u = self.sa_strategy.do_adapt(self.ts, self.dispatcher, self.workers, job, pd, pmd)
                self.next_mapek_ts += self.sa_period

                 # 记录数据 跳过第一个y，跳过最后一个u。或者直接删除
                 # 在每轮适应结束后记录
                 # 记录数据 每次添加的都是上一轮u的系统输出y，所以应该错位
                
                if self.flag:
                    pass
                    self.flag = False
                else:
                    self.y = np.array([[pd], [pmd]])
                    self.y_data_list.append(self.y)
                self.u_data_list.append(self.u)
                
                
                
                """self.y = np.array([[pd], [pmd]])
                self.u_data_list.append(self.u)
                self.y_data_list.append(self.y)"""





                


        self.ts = ts
        self.__advance_time_in_workers()

        # 每次输入一个job都会调用一次，
        # 由谁来调用？
        # main模块
    def run(self, job):
        """Advance the simulation up to the point when new job is being spawned and add it to the queues.
        
        The simulation may perform many internal steps (e.g., invoke MAPE-K multiple times) in one run invocation.
        If job is None, the run will perform final steps and conclude the simulation.
        """
        # 在两次作业提交的间隔可能有很久，按照特定的适应间隔，会执行很多次适应，但这是合理的，因为workerqueue里并不一定是空的，
        # 可能有很多其他的任务没有完成，所以依然要不断地执行适应
        # first run, initialize simulation
        if self.ts == 0.0:
            self.__start_simulation(job.spawn_ts)
            self.flag = True

        if job:
            # regular simulation step 运行到下一个任务生成时
            # advance_time方法会自动执行适应周期，直至最新的任务生成，其间没有任务到来也要适应
            self.__advance_time(job.spawn_ts)
            """    
            # 生成metric
            pd = self.metrics[0].get_period_delay      # job delay
            pmd = self.metrics[0].get_period_max_delay
            pw = self.metrics[1]     # power

            self.metrics[0].get_jobs
            if self.sa_strategy:  # run SA out of order (instantly, just before a job is dispatched)
                self.u = self.sa_strategy.do_adapt(self.ts, self.dispatcher, self.workers, job, pd, pmd)"""
            self.dispatcher.dispatch(job, self.workers)

        else: # 此时开始就没有job 进入了，是最末尾的时刻，需要慢慢把之前处理的任务处理完
            # 这个时间点 已经没有job到来了，不用再更新u了
            # let's wrap up the simulation 更新到最晚结束时间
            # 
            end_ts = self.ts
            for worker in self.workers:
                worker_end_ts = worker.get_finish_ts()
                if worker_end_ts:
                    end_ts = max(end_ts, worker.get_finish_ts())
            self.__advance_time(end_ts + self.sa_period) # 保证了最后一轮适应一定发生在 最后一个job结束前

            # 直接计算矩阵并打印出来
            # 真的不需要再进行一轮数据收集吗？
            model_order = 3
            # identified_model = si(self.y_data_list, self.u_data_list, dt=10, model_order=model_order, algorithm='n4sid')
                # 输出辨识得到的模型参数
"""            print("系统辨识结果：")
            print("A矩阵：\n", identified_model.A)
            print("B矩阵：\n", identified_model.B)
            print("C矩阵：\n", identified_model.C)
            print("D矩阵：\n", identified_model.D)"""

    






        
        






