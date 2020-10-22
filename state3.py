from definitions import Cube
import func

def collect(cube, cache, depth=1):
    for op in func.get_operations(3):
        _cube = cube.copy()
        _cube.apply_operation(op)
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
    func.dump('state3-steps.pkl', cache)
    print(f'{len(cache)} records dumped')

def verify_state3():
    import numpy as np
    cache = func.load('state3.pkl')
    assert cache[Cube().hash] == 0
    for _ in range(16):
        cube = Cube()
        seq = func.random_operations(3, np.random.randint(0, 21))
        cube.apply_operations(seq)
        print(len(seq), cache[cube.hash])

###########################################################

def collect_code(state3, cube, cache, depth=1):
    identity = Cube()
    for op in func.get_operations(3):
        _cube = cube.copy()
        _cube.apply_operation(op)
        if depth != state3[_cube.hash]: continue
        if _cube.hash in cache:
            continue
        cache[_cube.hash] = func.compute_relative_code(identity, _cube)
        if len(cache) % 100000 == 0:
            print(f'{len(cache)}')
        if depth < 15:
            collect_code(state3, _cube, cache, depth+1)

def dump_state3_code():
    state3 = func.load('state3.pkl')
    cache = {Cube().hash: func.compute_relative_code(Cube(), Cube())}
    collect_code(state3, Cube(), cache)
    func.dump('state3-codes.pkl', cache)
    print(f'{len(cache)} records dumped')

if __name__ == '__main__':
    #dump_state3_code()
    #dump_state3()
    verify_state3()
