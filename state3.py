from definitions import Cube
import func

def collect(cube, cache, depth=0):
    for op in func.get_operations(3):
        _cube = cube.copy()
        func.apply_operation(_cube, op)
        if _cube.hash in cache:
            continue
        cache.add(_cube.hash)
        if depth < 100:
            collect(_cube, cache, depth+1)

if __name__ == '__main__':
    cube = Cube()
    cache = set()
    collect(cube, cache)
    import pickle
    with open('state3.pkl', 'wb') as file:
        pickle.dump(cache, file)
    print(len(cache))
