from definitions import Cube
import func

def collect(cube, cache, depth=0):
    for op in func.get_operations(3):
        if depth == 0:
            print(f'computing {op}')
        _cube = cube.copy()
        func.apply_operation(_cube, op)
        if _cube.hash in cache and cache[_cube.hash] <= depth:
            continue
        cache[_cube.hash] = depth
        if len(cache) % 100000 == 0:
            print(f'{len(cache)}')
        if depth < 14:
            collect(_cube, cache, depth+1)

if __name__ == '__main__':
    cube = Cube()
    cache = {cube.hash: 0}
    collect(cube, cache)
    import pickle
    with open('state3.pkl', 'wb') as file:
        pickle.dump(cache, file)
    print(len(cache))
