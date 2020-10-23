import torch, os, pickle, time, random
import numpy as np
from definitions import Cube
from collections import defaultdict

def flatten(l2):
    return [item for l1 in l2 for item in l1]

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def ensure_folder(filename):
    folder = os.path.dirname(os.path.abspath(filename))
    mkdir(folder)
    return folder

def dump(path, obj):
    ensure_folder(path)
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def load(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

STATE3 = load('state3-steps.pkl')
def make_cube(label):
    if label == 4:
        return Cube()
    def _sample(operations, sizes=[1,3,5,7,8,9,10,11,12,13,14,15]):
        size = random.choice(sizes)
        return random.choices(operations, k=size)
    def _c3_sequence():
        length = np.random.randint(0, 16)
        operations = random.choices(get_operations(3), k=length)
        return operations
    def _c2_sequence():
        operations = _c3_sequence() if np.random.ranf() < 0.99 else []
        operations += _sample(['U','D'])
        return operations
    def _c1_sequence():
        operations = _c2_sequence() if np.random.ranf() < 0.99 else []
        operations += _sample(['F','B'])
        return operations
    def _c0_sequence():
        operations = _c1_sequence() if np.random.ranf() < 0.99 else []
        operations += _sample(['L','R'])
        return operations
    operations = [_c0_sequence, _c1_sequence, _c2_sequence, _c3_sequence]
    while True:
        seq = operations[label]()
        np.random.shuffle(seq)
        cube = Cube()
        cube.apply_operations(seq)
        if cube.hash == Cube.FINALE:
            continue
        if label < 3 and cube.hash in STATE3:
            continue
        else:
            break
    return cube


def _face_positions():
    '''
    _positions = [
        (0,29,36), (1,28), (2,27,47), (3,37), (4,), (5,46), (6,18,38), (7,19), (8,20,45),
        (32,39), (31,), (30,50), (40,), (49,), (21,41), (22,), (23,48),
        (15,35,42), (16,34), (17,33,53), (12,43), (13,), (14,52), (9,24,44), (10,25), (11,26,51)]
    '''
    _positions = [
        [0,29,36], [1,28], [2,27,47], [3,37], [5,46], [6,18,38], [7,19], [8,20,45],
        [32,39], [30,50], [21,41], [23,48],
        [15,35,42], [16,34], [17,33,53], [12,43], [14,52], [9,24,44], [10,25], [11,26,51]
    ]
    return _positions

def _component_colors(cube):
    data = cube.numpy()
    colors = [sorted(data[x]) for x in _face_positions()]
    return colors

def compute_relative_code(cube0, cube1):
    data = cube0.numpy()
    colors = _component_colors(cube1)
    codes = []
    for grid in _face_positions():
        grid_colors = sorted(data[grid])
        idx = colors.index(grid_colors)
        codes.append(idx)
    return tuple(codes)

def save_model(model):
    ckpt = {
        'model': model.state_dict()
    }
    path = 'checkpoints/model.pt'
    ensure_folder(path)
    torch.save(ckpt, path)

def load_model():
    from models import Model
    model = Model()
    path = 'checkpoints/model.pt'
    if os.path.isfile(path):
        ckpt = torch.load(path, map_location=lambda storage, location: storage)
        model.load_state_dict(ckpt['model'])
    return model.to(0)

def has_checkpoint():
    path = 'checkpoints/model.pt'
    return os.path.isfile(path)

def save_code_model(model):
    ckpt = {
        'model': model.state_dict()
    }
    path = 'checkpoints/code-model.pt'
    ensure_folder(path)
    torch.save(ckpt, path)

def load_code_model():
    from models import CodeModel
    model = CodeModel()
    path = 'checkpoints/code-model.pt'
    if os.path.isfile(path):
        ckpt = torch.load(path, map_location=lambda storage, location: storage)
        model.load_state_dict(ckpt['model'])
    return model.to(0)

def has_code_checkpoint():
    path = 'checkpoints/code-model.pt'
    return os.path.isfile(path)


def get_operations(level):
    operations = {
        0: ['U', 'D', 'L', 'R', 'F', 'B'],
        1: ['U', 'U2', 'U3', 'D', 'D2', 'D3', 'L2', 'R2', 'F', 'F2', 'F3', 'B', 'B2', 'B3'],
        2: ['U', 'U2', 'U3', 'D', 'D2', 'D3', 'L2', 'R2', 'F2', 'B2'],
        3: ['U2', 'D2', 'L2', 'R2', 'F2', 'B2']
    }
    return operations[level]

def random_operations(level, size):
    if level == 4:
        return []
    operations = get_operations(level)
    return np.random.choice(operations, size)

def solve(cube, search_method):
    from models import Model
    model = load_model()
    model.eval()
    cube = cube.copy()
    level = 0
    with torch.no_grad():
        sequence = []
        while True:
            level = model.predict([cube], level, 8)[0]
            print(f'current level: {level}')
            timer = Timer()
            if level == 4:
                break
            _sequence = search_method(model, cube, level)
            sequence += _sequence
            cube.apply_operations(_sequence)
            print(f'[{timer.check():>.2F}] level {level} solved {_sequence}.')
    return sequence

class Timer:
    __TIMERS__ = defaultdict(lambda: 0)
    __TOTAL__ = defaultdict(lambda: 0)

    def __init__(self):
        self._time = time.time()

    def check(self, name=None, count=None):
        now = time.time()
        elapsed = now - self._time
        self._time = now
        if name is not None:
            __class__.__TIMERS__[name] += elapsed
            __class__.__TOTAL__[name] += count or 0
        return elapsed

    @staticmethod
    def print():
        for k, v in __class__.__TIMERS__.items():
            c = __class__.__TOTAL__[k]
            if c == 0:
                print(f'{k}: {v:>.2F}')
            else:
                print(f'{k}: {v:>.2F}/{c}={v/c}')

if __name__ == '__main__':
    cube0 = Cube()
    cube0.apply_operations(['L2', 'U2', 'D', 'R', 'B', 'R2', 'F'])
    cube1 = cube0.copy()
    cube1.apply_operations(['U', 'D', 'L2', 'R2', 'L2', 'F2', 'B2'])
    code = compute_relative_code(cube0, cube1)
    codes = load('code-steps.pkl')
    print(codes[code])