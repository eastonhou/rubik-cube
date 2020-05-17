import func
from definitions import Cube

if __name__ == '__main__':
    cube = Cube([
        list('booywrgob'),
        list('gwyyybrby'),
        list('owwwrbrgo'),
        list('ygogorrob'),
        list('wowggbyrw'),
        list('rwbybygrg')
    ])
    func.apply_operations(cube, ['B','R','U','F','F','R'])
    func.apply_operations(cube, ['D','L2','D','R2','L2','B','F','L2','B','R2','U','B'])
    func.apply_operations(cube, ['U','D','D','B2','R2','U','B2','L2','U','D','B2','U','B2','U','R2'])
    func.apply_operations(cube, ['L2', 'U2', 'L2', 'B2', 'R2', 'D2', 'F2', 'R2'])
    if cube.hash == Cube.FINALE:
        print('already done')
    else:
        assert cube.validate()
        sequence = func.solve(cube, [0, 3])
        print(sequence)
