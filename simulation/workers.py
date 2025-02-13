
class WorkerQueue:
    """Main abstraction that represents jobs waiting for a particular worker.

    Queues process jobs in FIFO manner and once job is enqueued, it will wait until it is processed.
    A queue is also decorated by generic attributes, which are interpreted by the job Dispatcher.
    """
# 初始化里定义了 wokerqueue里面有的
    def __init__(self, **attributes):
        """The constructor gets initial attributes as named parameters."""
        self.jobs = []
        self.attributes = attributes

    def get_attribute(self, name):
        """Safe getter that returns attribute of given name or None if the attribute does not exist."""
        if name not in self.attributes:
            return None
        return self.attributes[name]


    def set_attribute(self, name, value):
        """Setter for attributes. This method is expected to be used by self-adapting algorithm."""
        self.attributes[name] = value

    def jobs_count(self):
        """Length of the queue."""
        return len(self.jobs)

    def get_finish_ts(self):#ts:time stamp,时间戳，获取队列最后一个任务的完成时间,以此来比较哪个队列更短
        """Get finish timestamp of last job in the queue, None if the queue is empty."""
        return self.jobs[-1].finish_ts if self.jobs else None

    # Methods used by the simulation to manage jobs
    # job[-1]表示获取队列最后一个任务
    # 后面一个迷你函数表示，如果队列不为空，则返回队尾job，如果为空则返回none
    def enqueue(self, job):
        """Place another job at the end of the queue."""
        job.enqueue(self.jobs[-1] if self.jobs else None)
        self.jobs.append(job)

# ts是目标时间，由外界传导进来的参数（快进到下一次mape-k适应，或者下一个任务到来）
    def advance_time(self, ts):
        """Advance time to a certain timestamp. Finished jobs are removed from the queue and returned."""
        res = []
        if self.jobs:
            i = 0
            while i < len(self.jobs) and self.jobs[i].finish_ts <= ts:
                i += 1
            res = self.jobs[:i]  # jobs that are finished (after ts)
            self.jobs = self.jobs[i:]  # remaining jobs

        return res
    
