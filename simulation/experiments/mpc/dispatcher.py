
from interfaces import AbstractDispatcher
from jobs import JobDurationIndex



class mpcDispatcher(AbstractDispatcher):
    
    def init(self, ts, workers):
        pass

    def dispatch(self, job, workers):
        active_workers = list(filter(lambda w: w.get_attribute("active"), workers))
        if not active_workers:
            raise RuntimeError("No active workers available, unable to dispatch job.")

#找到最早完成任务（fin.ts最小）的worker
        active_workers.sort(key=lambda x: x.get_finish_ts())
        target = active_workers[0]
        target.enqueue(job)




















