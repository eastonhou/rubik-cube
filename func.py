import torch, os, pickle, time
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

STATE3 = load('state3.pkl')
def make_cube(label):
    def _sample(operations, sizes=[1,3,5,7,8,9,10,11,12,13,14,15]):
        size = np.random.choice(sizes)
        return np.random.choice(operations, size).tolist()
    def _c3_sequence():
        length = np.random.randint(0, 16)
        operations = np.random.choice(get_operations(3), length).tolist()
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
        apply_operations(cube, seq)
        if label < 3 and cube.hash in STATE3:
            continue
        else:
            break
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
            if level == 4:
                break
            _sequence = search_method(model, cube, level)
            sequence += _sequence
            apply_operations(cube, _sequence)
            print(f'level {level} solved {_sequence}.')
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
