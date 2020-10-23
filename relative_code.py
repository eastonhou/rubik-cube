import func
import numpy as np
from definitions import Cube
from collections import defaultdict
from asynchronous import DataProducer

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
        self.timer = func.Timer()
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
        print(
            f'[{self.timer.check():>.2F}]'
            f' {len(self.cache)}:{sum(self.cache.values())}'
            f' level={self.level} depth={self.depth}')

def collect(maxdepth):
    identity = Cube()
    producer = Producer(identity, 2, maxdepth)
    cubes = {}
    while True:
        records = producer.get(512)
        if not records: break
        for seq, cube in records:
            if cube.hash not in cubes:
                code = func.compute_relative_code(identity, cube)
                cubes[cube.hash] = code, len(seq)
            else:
                code, distance = cubes[cube.hash]
                cubes[cube.hash] = code, min(len(seq), distance)
    producer.cancel()
    producer.print_summary()
    result = {x:y for x,y in cubes.values()}
    assert len(result) == len(cubes)
    result[func.compute_relative_code(identity, identity)] = 0
    return result

if __name__ == '__main__':
    cubes = collect(7)
    func.dump('code-steps.pkl', cubes)
