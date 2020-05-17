import numpy as np

def _hash(value):
    return hash(tuple(value))

class Cube:
    FINALE = _hash(np.reshape(np.tile(np.arange(6), (9,1)).transpose(), -1))
    @staticmethod
    def from_data(data):
        cmap = {k:v for v,k in enumerate('wyrogb')}
        layer0 = np.array([[cmap[x] for x in y] for y in data], dtype=np.uint8)
        layer1 = np.tile(np.full(9, 4, dtype=np.uint8), (6,1))
        cube = Cube()
        cube._data = np.reshape(np.stack((layer0, layer1), axis=-1), (-1,2))
        return cube

    def to_data(self):
        cmap = np.array(list('wyrogb'))
        nmap = np.array(list('012345678'))
        layer0, layer1 = cmap[self._data[...,0]], nmap[self._data[...,1]]
        result = np.reshape(np.core.defchararray.add(layer0, layer1), (6,-1))
        return result

    def __init__(self, data=None):
        if data is None:
            self.reset()
        else:
            self._data = data
            self._hash = None

    def copy(self):
        return Cube(self._data)

    @property
    def hash(self):
        if self._hash is None:
            self._hash = _hash(self.numpy())
        return self._hash

    def numpy(self):
        return np.reshape(self._data[...,0], -1)

    def reset(self):
        layer0 = np.tile(np.arange(6, dtype=np.uint8), (9,1)).transpose()
        layer1 = np.tile(np.arange(9, dtype=np.uint8), (6,1))
        self._data = np.reshape(np.stack((layer0, layer1), axis=-1), (-1,2))
        self._hash = None

    __FMAP__ = np.array([
        0,1,2,3,4,5,45,48,51,
        38,41,44,12,13,14,15,16,17,
        20,23,26,19,22,25,18,21,24,
        27,28,29,30,31,32,33,34,35,
        36,37,8,39,40,7,42,43,6,
        11,46,47,10,49,50,9,52,53])
    def rotate_front(self):
        return self._apply_map(__class__.__FMAP__)

    __BMAP__ = np.array([
        42,39,36,3,4,5,6,7,8,
        9,10,11,12,13,14,53,50,47,
        18,19,20,21,22,23,24,25,26,
        29,32,35,28,31,34,27,30,33,
        15,37,38,16,40,41,17,43,44,
        45,46,0,48,49,1,51,52,2])
    def rotate_back(self):
        self._apply_map(__class__.__BMAP__)

    __UMAP__ = np.array([
        2,5,8,1,4,7,0,3,6,
        9,10,11,12,13,14,15,16,17,
        36,37,38,21,22,23,24,25,26,
        45,46,47,30,31,32,33,34,35,
        27,28,29,39,40,41,42,43,44,
        18,19,20,48,49,50,51,52,53])
    def rotate_top(self):
        self._apply_map(__class__.__UMAP__)

    __DMAP__ = np.array([
        0,1,2,3,4,5,6,7,8,
        11,14,17,10,13,16,9,12,15,
        18,19,20,21,22,23,51,52,53,
        27,28,29,30,31,32,42,43,44,
        36,37,38,39,40,41,24,25,26,
        45,46,47,48,49,50,33,34,35])
    def rotate_bottom(self):
        self._apply_map(__class__.__DMAP__)

    __LMAP__ = np.array([
        18,1,2,21,4,5,24,7,8,
        35,10,11,32,13,14,29,16,17,
        9,19,20,12,22,23,15,25,26,
        27,28,6,30,31,3,33,34,0,
        38,41,44,37,40,43,36,39,42,
        45,46,47,48,49,50,51,52,53])
    def rotate_left(self):
        self._apply_map(__class__.__LMAP__)

    __RMAP__ = np.array([
        0,1,33,3,4,30,6,7,27,
        9,10,20,12,13,23,15,16,26,
        18,19,2,21,22,5,24,25,8,
        17,28,29,14,31,32,11,34,35,
        36,37,38,39,40,41,42,43,44,
        47,50,53,46,49,52,45,48,51])
    def rotate_right(self):
        self._apply_map(__class__.__RMAP__)

    def face(self, n, layer=0):
        return self._data[n,:,layer]

    __UMAP2__ = __UMAP__[__UMAP__]
    __DMAP2__ = __DMAP__[__DMAP__]
    __FMAP2__ = __FMAP__[__FMAP__]
    __BMAP2__ = __BMAP__[__BMAP__]
    __LMAP2__ = __LMAP__[__LMAP__]
    __RMAP2__ = __RMAP__[__RMAP__]
    __OPERATIONS__ = {
        'U': __UMAP__, 'D': __DMAP__,
        'F': __FMAP__, 'B': __BMAP__,
        'L': __LMAP__, 'R': __RMAP__,
        'U2': __UMAP2__, 'D2': __DMAP2__,
        'F2': __FMAP2__, 'B2': __BMAP2__,
        'L2': __LMAP2__, 'R2': __RMAP2__,
    }
    def apply_operation(self, operation):
        self._apply_map(__class__.__OPERATIONS__[operation])

    def _apply_map(self, imap):
        self._data = self._data[imap,:]
        self._hash = None

    def validate(self):
        colors, counts = np.unique(self._data[...,0], return_counts=True)
        result = np.all(colors==np.arange(6)) and np.all(counts==9)
        return result

    def print(self):
        print(self.to_data())