import threading, time
import numpy as np

class DataProducer:
    def __init__(self, minsize, maxsize):
        self.queue = []
        self.lock = threading.Lock()
        self.minsize = minsize
        self.maxsize = maxsize

    def start(self):
        self.stop_flag = False
        self.thread = run_async(self._worker)

    def _worker(self):
        while not self.stop_flag:
            if self.size >= self.maxsize:
                time.sleep(1)
                continue
            self._produce()

    def _produce(self):
        raise NotImplementedError()

    @property
    def size(self):
        return len(self.queue)

    def get(self, size):
        while self.size < self.minsize and not self.stop_flag:
            time.sleep(0.1)
        self.lock.acquire()
        records = self.queue[:size]
        self.queue = self.queue[size:]
        self.lock.release()
        return records

    def put(self, records):
        while self.size >= self.maxsize:
            if self.stop_flag:
                return
            time.sleep(0.1)
        self.lock.acquire()
        self.queue += records
        self.lock.release()

    def cancel(self):
        self.stop_flag = True
        self.thread.join()

def run_async(target, *args):
    thread = threading.Thread(target=target, args=args)
    thread.start()
    return thread
