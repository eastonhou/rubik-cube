import threading, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class DataProducer:
    def __init__(self, minsize, maxsize):
        self.queue = []
        self.lock = threading.Lock()
        self.minsize = minsize
        self.maxsize = maxsize

    def start(self):
        self.stop_flag = False
        run_async(self._worker)

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
        self.lock.acquire()
        self.queue += records
        self.lock.release()

    def cancel(self):
        self.stop_flag = True

def run_async(target, *args):
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(target, *args)
