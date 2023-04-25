from ray.util.queue import Queue

class Storage(object):
    def __init__(self, threshold=15, size=20):
        self.threshold = threshold
        self.batch_queue = Queue(maxsize=size)
        self.trajectory_queue = Queue(maxsize=size)
        self.priority_queue = Queue(maxsize=size)

    def push_batch(self, batch):
        if self.batch_queue.qsize() <= self.threshold:
            self.batch_queue.put(batch)

    def pop_batch(self):
        if self.batch_queue.qsize() > 0:
            return self.batch_queue.get()
        else:
            return None

    def push_trajectory(self, trajectory):
        if self.trajectory_queue.qsize() <= self.threshold:
            self.trajectory_queue.put(trajectory)

    def pop_trajectory(self):
        if self.trajectory_queue.qsize() > 0:
            return self.trajectory_queue.get()
        else:
            return None

    def push_priority(self, priority):
        if self.priority_queue.qsize() <= self.threshold:
            self.priority_queue.put(priority)

    def pop_priority(self):
        if self.priority_queue.qsize() > 0:
            return self.priority_queue.get()
        else:
            return None


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)