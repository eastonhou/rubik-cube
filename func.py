import torch, os
import numpy as np
from definitions import Cube

def flatten(l2):
    return [item for l1 in l2 for item in l1]

def make_cube(label):
    def _sample(operations, base):
        size = np.random.randint(base, 20)
        return np.random.choice(operations, size).tolist()
    def _c3_sequence(base):
        length = np.random.randint(base, 50)
        operations = np.random.choice(get_operations(3), length).tolist()
        return operations
    def _c2_sequence(base):
        operations = _c3_sequence(20)
        operations += _sample(['U','D'], base)
        return operations
    def _c1_sequence(base):
        operations = _c2_sequence(10)
        operations += _sample(['F','B'], base)
        return operations
    def _c0_sequence(base):
        operations = _c1_sequence(10)
        operations += _sample(['L','R'], base)
        return operations
    operations = [_c0_sequence, _c1_sequence, _c2_sequence, _c3_sequence]
    seq = operations[label](2)
    np.random.shuffle(seq)
    cube = Cube()
    for op in seq:
        apply_operation(cube, op)
    return cube

def apply_operation(cube, operation):
    operations = {
        'U': cube.rotate_top, 'D': cube.rotate_bottom,
        'F': cube.rotate_front, 'B': cube.rotate_back,
        'L': cube.rotate_left, 'R': cube.rotate_right
    }
    for _ in range(len(operation)):
        operations[operation[0]]()

def apply_operations(cube, operations):
    for op in operations:
        #cube0 = cube.copy()
        apply_operation(cube, op)
        #if False == cube.validate():
        #    apply_operation(cube0, op)
        #    assert False

def save_model(model):
    ckpt = {
        'model': model.state_dict()
    }
    folder = 'checkpoints'
    path = os.path.join(folder, 'model.pt')
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(ckpt, path)

def load_model():
    from models import Model
    model = Model()
    path = 'checkpoints/model.pt'
    if os.path.isfile(path):
        ckpt = torch.load(path, map_location=lambda storage, location: storage)
        model.load_state_dict(ckpt['model'])
    return model.to(0)

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
        if _last_op == _op[0] and _last_count + len(_op) >= 4:
            continue
        else:
            _cube = cube.copy()
            apply_operation(_cube, _op)
            result.append((_op, _cube))
    return result

def get_operations(level):
    operations = {
        0: ['U', 'D', 'L', 'R', 'F', 'B'],
        1: ['U', 'D', 'L2', 'R2', 'F', 'B'],
        2: ['U', 'D', 'L2', 'R2', 'F2', 'B2'],
        3: ['U2', 'D2', 'L2', 'R2', 'F2', 'B2'],
    }
    return operations[level]

def random_operations(level, size):
    operations = get_operations(level)
    return np.random.choice(operations, size)

def search(model, cube, level, sequence, maxdepth, cache):
    if len(sequence) >= maxdepth:
        return None
    _ops, _cubes = zip(*_valid_operations(cube, sequence, get_operations(level)))
    _levels = model.predict(_cubes, level)
    for _op, _cube, _level in zip(_ops, _cubes, _levels):
        if _level > level:
            return sequence+[_op]
    for _op, _cube in zip(_ops, _cubes):
        if _cube.hash in cache and cache[_cube.hash] >= maxdepth-len(sequence):
            continue
        cache[_cube.hash] = maxdepth-len(sequence)
        _sequence = sequence+[_op]
        _sequence = search(model, _cube, level, _sequence, maxdepth, cache)
        if _sequence is not None:
            return _sequence
    return None

def search2(model, cube, level):
    maxdepths = [8, 20, 40, 60]
    maxdepth = maxdepths[level]
    for _ in range(10000000):
        ops, cubes = [], []
        for _ in range(256):
            _ops = random_operations(level, np.random.randint(1, maxdepth))
            _cube = cube.copy()
            apply_operations(_cube, _ops)
            ops.append(_ops)
            cubes.append(_cube)
        levels = model.predict(cubes)
        if levels.max() > level:
            x = levels.argmax()
            return ops[x].tolist()
    return None

def solve(cube):
    from models import Model
    model = load_model()
    model.eval()
    cube = cube.copy()
    with torch.no_grad():
        sequence = []
        while True:
            level = model.predict([cube], 0, 16)[0]
            print(f'current level: {level}')
            if level == 4:
                break
            if level in [0, 1, 2, 3]:
                _cache = {}
                for maxdepth in range(1, 40):
                    _sequence = search(model, cube, level, [], maxdepth, _cache)
                    print(f'{len(_cache)} steps sought. level={level} depth={maxdepth}')
                    if _sequence is not None:
                        break
            else:
                _sequence = search2(model, cube, level)
            if _sequence is not None:
                sequence += _sequence
                for _op in _sequence:
                    apply_operation(cube, _op)
                assert cube.validate()
                print(f'level {level} solved [{",".join(_sequence)}].')
                print(f'==state: {["".join(x) for x in cube.data]}')
    return sequence

if __name__ == '__main__':
    #cube = make_cube(0)
    '''
    cube = Cube([
        list('wybwwwywy'),
        list('wgowyrwyb'),
        list('rgrbrboog'),
        list('rgbooryrr'),
        list('obbggygog'),
        list('grwobyybo')
    ])
    '''
    cube = Cube([
        list('booywrgob'),
        list('gwyyybrby'),
        list('owwwrbrgo'),
        list('ygogorrob'),
        list('wowggbyrw'),
        list('rwbybygrg')
    ])
    apply_operations(cube, ['B','R','U','F','F','R'])
    apply_operations(cube, ['L2','B','L2','B','L2','B','U','R2','D','L2','B'])
    apply_operations(cube, ['U','R2','B2','U','D','B2','U'])
    #apply_operations(cube, ['L2','B2','B2','U','L2','L2','F2','R2','D','B2','D','B2','B2','F2','R2','D','F2'])
    assert cube.validate()
    sequence = solve(cube)
    for op in sequence:
        apply_operation(cube, op)
