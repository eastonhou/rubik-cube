from definitions import Cube
import func

def collect(cube, cache, depth=1):
    for op in func.get_operations(3):
        _cube = cube.copy()
        func.apply_operation(_cube, op)
        if _cube.hash in cache and cache[_cube.hash] <= depth:
            continue
        cache[_cube.hash] = depth
        if len(cache) % 100000 == 0:
            print(f'{len(cache)}')
        if depth < 15:
            collect(_cube, cache, depth+1)

def dump_state3():
    cube = Cube()
    cache = {cube.hash: 0}
    collect(cube, cache)
    func.dump('state3.pkl', cache)
    operations = func.random_operations(3, 21)
    func.apply_operations(cube, operations)

def verify_state3():
    import numpy as np
    cache = func.load('state3.pkl')
    cube = Cube()
    assert cache[cube.hash] == 0
    for _ in range(16):
        seq = func.random_operations(3, np.random.randint(0, 21))
        func.apply_operations(cube, seq)
        print(len(seq), cache[cube.hash])

if __name__ == '__main__':
    #dump_state3()
    verify_state3()