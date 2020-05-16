import torch, os
import numpy as np
from definitions import Cube

def flatten(l2):
    return [item for l1 in l2 for item in l1]

def make_cube(label):
    def _pair(nonzero):
        r = np.random.randint(1 if nonzero else 0, 4)
        return r//2, r%2
    def _c3_sequence(nonzero):
        length = np.random.randint(0, 20)
        operations = np.random.choice(get_operations(3), length).tolist()
        return operations
    def _c2_sequence(nonzero):
        operations = _c3_sequence(False)
        u, d = _pair(nonzero)
        if u:
            operations.append('U')
        if d:
            operations.append('D')
        return operations
    def _c1_sequence(nonzero):
        operations = _c2_sequence(False)
        f, b = _pair(nonzero)
        if f:
            operations.append('F')
        if b:
            operations.append('B')
        return operations
    def _c0_sequence(nonzero):
        operations = _c1_sequence(False)
        l, r = _pair(nonzero)
        if l:
            operations.append('L')
        if r:
            operations.append('R')
        return operations
    operations = [_c0_sequence, _c1_sequence, _c2_sequence, _c3_sequence]
    seq = operations[label](True)
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

_FINALE = flatten(Cube().data)
def predict(model, cube):
    if flatten(cube.data) == _FINALE:
        return 4
    level = model([cube])[0].argmax(-1).item()
    return level

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

def search(model, cube, level, sequence):
    if len(sequence) >= 10:
        return None
    _ops, _cubes = zip(*_valid_operations(cube, sequence, get_operations(level)))
    _levels = model.predict(_cubes)
    for _op, _cube, _level in zip(_ops, _cubes, _levels):
        if level == 3 and flatten(cube.data) == _FINALE:
            return sequence+[_op]
        if _level > level:
            return sequence+[_op]
    for _op, _cube in zip(_ops, _cubes):
        _sequence = sequence+[_op]
        _sequence = search(model, _cube, level, _sequence)
        if _sequence is not None:
            return _sequence
    return None

def solve(cube):
    from models import Model
    model = load_model()
    model.eval()
    cube = cube.copy()
    with torch.no_grad():
        sequence = []
        while True:
            level = predict(model, cube)
            if level == 4:
                break
            _sequence = search(model, cube, level, [])
            if _sequence is not None:
                sequence += _sequence
                for _op in _sequence:
                    apply_operation(cube, _op)
                print(f'level {level} solved [{"".join(_sequence)}].')
    return sequence

if __name__ == '__main__':
    cube = make_cube(0)
    sequence = solve(cube)
    for op in sequence:
        apply_operation(cube, op)
    level = predict(None, cube)
    assert level == 4
