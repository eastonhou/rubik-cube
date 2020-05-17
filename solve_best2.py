import func
import numpy as np
from models import Model
from definitions import Cube
from asynchronous import DataProducer
from collections import defaultdict

def _determine_last_operations(seq):
    if len(seq)>=3 and seq[-3] == seq[-2] == seq[-1]:
        return seq[-1], 3
    if len(seq)>=2 and seq[-1][0] == seq[-2][0]:
        return seq[-1][0], len(seq[-1])+len(seq[-2])
    if len(seq)>=1:
        return seq[-1][0], len(seq[-1])
    return 'X', 0

def _valid_operations(cube, sequence, operations):
    _operations = operations.copy()
    np.random.shuffle(_operations)
    result = []
    for _op in _operations:
        _last_op, _last_count = _determine_last_operations(sequence)
        if _last_op == _op[0] and _last_count + len(_op) >= 4:
            continue
        else:
            _cube = cube.copy()
            _cube.apply_operation(_op)
            result.append((_op, _cube))
    return result

class Producer(DataProducer):
    def __init__(self, cube, level, depth):
        super(__class__, self).__init__(1, 256*100)
        self.level = level
        self.depth = depth
        self.cube = cube
        self.cache = defaultdict(lambda: -1)
        self.start()

    def _iterate(self, cube, sequence):
        operations = func.get_operations(self.level)
        _ops, _cubes = zip(*_valid_operations(cube, sequence, operations))
        for _op, _cube in zip(_ops, _cubes):
            if self.cache[_cube.hash] >= self.depth-len(sequence):
                continue
            if self.stop_flag:
                return
            self.cache[_cube.hash] = self.depth-len(sequence)
            _sequence = sequence+[_op]
            self.put([(_sequence, _cube)])
            if len(_sequence) < self.depth:
                self._iterate(_cube, _sequence)

    def _worker(self):
        self._iterate(self.cube, [])
        self.stop_flag = True

    def print_summary(self):
        print(f'{len(self.cache)}:{sum(self.cache.values())} level={self.level} depth={self.depth}')

def _search(model, cube, level, maxdepth):
    producer = Producer(cube, level, maxdepth)
    result = None
    while True:
        records = producer.get(256)
        if not records:
            break
        seqs, cubes = zip(*records)
        levels = model.predict(cubes, level)
        idx = levels.argmax()
        if levels[idx] > level:
            result = seqs[idx]
            break
    producer.cancel()
    producer.print_summary()
    return result

def search(model, cube, level):
    for maxdepth in range(1, 17):
        _sequence = _search(model, cube, level, maxdepth)
        if _sequence is not None:
            return _sequence

if __name__ == '__main__':
    cube = Cube.from_data([
        'booywrgob',
        'gwyyybrby',
        'owwwrbrgo',
        'ygogorrob',
        'wowggbyrw',
        'rwbybygrg'
    ])
    func.solve(cube, search)
