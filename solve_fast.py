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
            _cube.apply_operations(_ops)
            ops.append(_ops)
            cubes.append(_cube)
        levels = model.predict(cubes, level)
        if levels.max() > level:
            x = levels.argmax()
            return ops[x].tolist()
    return None

if __name__ == '__main__':
    cube = Cube.from_data([
        'ywyywywwy',
        'wwyyywwyw',
        'bororogrr',
        'obororogo',
        'bbrggbbor',
        'grggbbbgg'
    ])
    cube.apply_operations(['R2', 'D', 'F2', 'U2', 'U'])
    seq = func.solve(cube, search)
    print(seq)
    cube.apply_operations(seq)
    cube.print()
    assert cube.hash == Cube.FINALE
