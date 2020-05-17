import func
import numpy as np
from definitions import Cube

def search(model, cube, level):
    maxdepths = [7, 13, 17, 19]
    maxdepth = maxdepths[level]
    for _ in range(10000000):
        ops, cubes = [], []
        for _ in range(256):
            _ops = func.random_operations(level, np.random.randint(1, maxdepth+1))
            _cube = cube.copy()
            func.apply_operations(_cube, _ops)
            ops.append(_ops)
            cubes.append(_cube)
        levels = model.predict(cubes, level)
        if levels.max() > level:
            x = levels.argmax()
            return ops[x].tolist()
    return None

if __name__ == '__main__':
    cube = Cube([
        list('booywrgob'),
        list('gwyyybrby'),
        list('owwwrbrgo'),
        list('ygogorrob'),
        list('wowggbyrw'),
        list('rwbybygrg')
    ])
    sequence = func.solve(cube, search)
    print(sequence)
