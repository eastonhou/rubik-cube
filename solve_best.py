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
    #apply_operations(cube, ['L2','B','L2','B','L2','B','U','R2','D','L2','B'])
    #apply_operations(cube, ['U','R2','B2','U','D','B2','U'])
    #assert cube.hash in load('state3.pkl')
    #apply_operations(cube, ['L2','B2','B2','U','L2','L2','F2','R2','D','B2','D','B2','B2','F2','R2','D','F2'])
    assert cube.validate()
    sequence = func.solve(cube, [0,1,2,3])
    print(sequence)
