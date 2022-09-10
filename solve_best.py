import func
import numpy as np
from definitions import Cube

def _determine_last_operations(seq):
    if len(seq)>=3 and seq[-3] == seq[-2] == seq[-1]:
        return seq[-1], 3
    if len(seq)>=2 and seq[-1][0] == seq[-2][0]:
        return seq[-1][0], len(seq[-1])+len(seq[-2])
    if len(seq)>=1:
        return seq[-1][0], len(seq[-1])
    return 'X', 0

def _valid_operations(cube, sequence, operations):
    _operations = np.random.choice(operations, 6, replace=False)
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

def _search(model, cube, level, sequence, maxdepth, cache):
    if len(sequence) >= maxdepth:
        return None
    _ops, _cubes = zip(*_valid_operations(cube, sequence, func.get_operations(level)))
    _levels = model.predict(_cubes, level)
    for _op, _cube, _level in zip(_ops, _cubes, _levels):
        if _level > level:
            return sequence+[_op]
    for _op, _cube in zip(_ops, _cubes):
        if _cube.hash in cache and cache[_cube.hash] >= maxdepth-len(sequence):
            continue
        cache[_cube.hash] = maxdepth-len(sequence)
        _sequence = sequence+[_op]
        _sequence = _search(model, _cube, level, _sequence, maxdepth, cache)
        if _sequence is not None:
            return _sequence
    return None

def search(model, cube, level):
    _cache = {}
    for maxdepth in range(1, 17):
        _sequence = _search(model, cube, level, [], maxdepth, _cache)
        print(f'{len(_cache)}:{sum(_cache.values())} level={level} depth={maxdepth}')
        if _sequence is not None:
            return _sequence

if __name__ == '__main__':
    cube = Cube.from_data([
        'yyyywwwww',
        'wywwyyywy',
        'rroorrroo',
        'rrorooroo',
        'bbgggbgbb',
        'ggbbbgbgg'
    ])
    #cube.apply_operation('R2')
    seq = func.solve(cube, search)
    print(seq)
    cube.apply_operations(seq)
    cube.print()
    assert cube.hash == Cube.FINALE
