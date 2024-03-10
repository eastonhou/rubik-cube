import func
import torch
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
        if _last_op == _op[0]:
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

def _search(model, cube, level, maxdepth):
    producer = Producer(cube, level, maxdepth)
    result = None
    while True:
        records = producer.get(512)
        if not records: break
        seqs, cubes = zip(*records)
        levels = model.predict(cubes, level)
        idx = levels.argmax()
        if levels[idx] > level:
            result = seqs[idx]
            break
    producer.cancel()
    producer.print_summary()
    return result

def _match_codes(identity, state3_codes, codes, cube):
    _code = func.compute_relative_code(cube, identity)
    _codes = state3_codes[:,_code]
    _codes2 = torch.cat((_codes, codes), dim=0)
    values, counts = _codes2.unique(return_counts=True, dim=0)
    return values[counts>1].cpu().numpy()

def _search_by_code_recursive(identity, state3_codes, codes, code_steps, cube, depth, minsteps):
    if depth >= 8:
        return None
    assert cube.hash != identity.hash

    _codes = _match_codes(identity, state3_codes, codes, cube)
    steps = [code_steps[tuple(x)] for x in _codes]
    if steps:
        _minidx = np.argmin(steps)
        _minsteps = steps[_minidx]
        if _minsteps >= minsteps:
            return None
        else:
            minsteps = _minsteps
            if 0 == minsteps:
                return []
    for op in func.get_operations(2):
        _cube = cube.copy()
        _cube.apply_operation(op)
        result = _search_by_code_recursive(identity, state3_codes, codes, code_steps, _cube, depth+1, minsteps)
        if result is not None:
            return [op] + result
    return None

def _search_by_code(cube, minsteps=100):
    identity = Cube()    
    #cube_code = func.compute_relative_code(cube, Cube())
    state3_codes = func.load('state3-codes.pkl')
    state3_codes = torch.tensor(tuple(state3_codes.values()), dtype=torch.uint8, device=0)
    code_steps = func.load('code-steps.pkl')
    codes = list(code_steps.keys())
    codes = torch.tensor(codes, dtype=torch.uint8, device=0)
    return _search_by_code_recursive(identity, state3_codes, codes, code_steps, cube, 0, 100)

def _cube_code_steps(state3_codes, code_steps, codes, cube):
    _codes = _match_codes(Cube(), state3_codes, codes, cube)
    steps = [code_steps[tuple(x)] for x in _codes]
    if steps:
        _minidx = np.argmin(steps)
        minsteps = steps[_minidx]
        return minsteps
    else:
        return None

def _search_by_code_recursive2(state3_codes, codes, code_steps, cube, minsteps):
    if minsteps == 0:
        return []
    for op in func.get_operations(2):
        _cube = cube.copy()
        _cube.apply_operation(op)
        _minsteps = _cube_code_steps(state3_codes, code_steps, codes, _cube)
        if _minsteps is not None and _minsteps < minsteps:
            return [op] + _search_by_code_recursive2(state3_codes, codes, code_steps, _cube, _minsteps)
    return None

def _search_by_code2(cube):
    code_model = func.load_code_model().eval()
    producer = Producer(cube, 2, 7)
    state3_codes = func.load('state3-codes.pkl')
    state3_codes = torch.tensor(tuple(state3_codes.values()), dtype=torch.uint8, device=0)
    code_steps = func.load('code-steps.pkl')
    codes = list(code_steps.keys())
    codes = torch.tensor(codes, dtype=torch.uint8, device=0)
    while True:
        records = producer.get(512)
        if not records: break
        seqs, cubes = zip(*records)
        predicts = code_model(cubes).argmax(-1).cpu().numpy()
        for seq, cube, predict in zip(seqs, cubes, predicts):
            if predict:
                minsteps = _cube_code_steps(state3_codes, code_steps, codes, cube)
                if minsteps is not None:
                    return seq + _search_by_code_recursive2(state3_codes, codes, code_steps, cube, minsteps)
    return None

def _search_by_state(cube):
    operations = []
    state3_steps = func.load('state3-steps.pkl')
    while cube.hash != Cube.FINALE:
        for op in func.get_operations(3):
            _cube = cube.copy()
            _cube.apply_operation(op)
            if state3_steps[_cube.hash] < state3_steps[cube.hash]:
                operations.append(op)
                cube = _cube
                break
    return operations

def search(model, cube, level):
    if level == 3:
        return _search_by_state(cube)
    elif level == 2:
        return _search_by_code2(cube)
    for maxdepth in range(1, 17):
        _sequence = _search(model, cube, level, maxdepth)
        if _sequence is not None:
            return _sequence

if __name__ == '__main__':
    cube = Cube.from_data([
        'ryyrwrowb',
        'obowywrwb',
        'woworyyob',
        'rbwooyyby',
        'ggggggggg',
        'rbbrbywro'
    ])
    #cube.apply_operations(['R2', 'D', 'F2', 'U', 'U2', 'R2', 'B2', 'U', 'F2', 'L2', 'U2', 'F2', 'R2', 'D', 'F2'])
    #cube = Cube()
    #cube.apply_operations(func.get_operations(2))
    seq = func.solve(cube, search)
    print(seq)
    cube.apply_operations(seq)
    cube.print()
    assert cube.hash == Cube.FINALE
