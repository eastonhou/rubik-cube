class Cube:
    FINALE = 'wwwwwwwwwyyyyyyyyyrrrrrrrrrooooooooogggggggggbbbbbbbbb'.__hash__()
    def __init__(self, data=None):
        if data is not None:
            self.data = data
            self._hash = None
        else:
            self.reset()

    def copy(self):
        cube = Cube([x for x in self.data])
        return cube

    @property
    def hash(self):
        if self._hash is None:
            from func import flatten
            self._hash = ''.join(flatten(self.data)).__hash__()
            #self._hash = ''.join(flatten(self.data))
        return self._hash

    def reset(self):
        '''
        self.data = [
            ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8'],
            ['y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8'],
            ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8'],
            ['o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8'],
            ['g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8'],
            ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']]
        '''
        self.data = [
            ['w']*9,
            ['y']*9,
            ['r']*9,
            ['o']*9,
            ['g']*9,
            ['b']*9
        ]
        self._hash = None

    def rotate_front(self):
        w, y, r, o, g, b = self.data
        self.data = [
            [w[0], w[1], w[2], w[3], w[4], w[5], b[0], b[3], b[6]],
            [g[2], g[5], g[8], y[3], y[4], y[5], y[6], y[7], y[8]],
            [r[2], r[5], r[8], r[1], r[4], r[7], r[0], r[3], r[6]],
            o,
            [g[0], g[1], w[8], g[3], g[4], w[7], g[6], g[7], w[6]],
            [y[2], b[1], b[2], y[1], b[4], b[5], y[0], b[7], b[8]]]
        self._hash = None

    def rotate_back(self):
        w, y, r, o, g, b = self.data
        self.data = [
            [g[6], g[3], g[0], w[3], w[4], w[5], w[6], w[7], w[8]],
            [y[0], y[1], y[2], y[3], y[4], y[5], b[8], b[5], b[2]],
            r,
            [o[2], o[5], o[8], o[1], o[4], o[7], o[0], o[3], o[6]],
            [y[6], g[1], g[2], y[7], g[4], g[5], y[8], g[7], g[8]],
            [b[0], b[1], w[0], b[3], b[4], w[1], b[6], b[7], w[2]]]
        self._hash = None

    def rotate_top(self):
        w, y, r, o, g, b = self.data
        self.data = [
            [w[2], w[5], w[8], w[1], w[4], w[7], w[0], w[3], w[6]],
            y,
            [g[0], g[1], g[2], r[3], r[4], r[5], r[6], r[7], r[8]],
            [b[0], b[1], b[2], o[3], o[4], o[5], o[6], o[7], o[8]],
            [o[0], o[1], o[2], g[3], g[4], g[5], g[6], g[7], g[8]],
            [r[0], r[1], r[2], b[3], b[4], b[5], b[6], b[7], b[8]]]
        self._hash = None

    def rotate_bottom(self):
        w, y, r, o, g, b = self.data
        self.data = [
            w,
            [y[2], y[5], y[8], y[1], y[4], y[7], y[0], y[3], y[6]],
            [r[0], r[1], r[2], r[3], r[4], r[5], b[6], b[7], b[8]],
            [o[0], o[1], o[2], o[3], o[4], o[5], g[6], g[7], g[8]],
            [g[0], g[1], g[2], g[3], g[4], g[5], r[6], r[7], r[8]],
            [b[0], b[1], b[2], b[3], b[4], b[5], o[6], o[7], o[8]]]
        self._hash = None

    def rotate_left(self):
        w, y, r, o, g, b = self.data
        self.data = [
            [r[0], w[1], w[2], r[3], w[4], w[5], r[6], w[7], w[8]],
            [o[8], y[1], y[2], o[5], y[4], y[5], o[2], y[7], y[8]],
            [y[0], r[1], r[2], y[3], r[4], r[5], y[6], r[7], r[8]],
            [o[0], o[1], w[6], o[3], o[4], w[3], o[6], o[7], w[0]],
            [g[2], g[5], g[8], g[1], g[4], g[7], g[0], g[3], g[6]],
            b]
        self._hash = None

    def rotate_right(self):
        w, y, r, o, g, b = self.data
        self.data = [
            [w[0], w[1], o[6], w[3], w[4], o[3], w[6], w[7], o[0]],
            [y[0], y[1], r[2], y[3], y[4], r[5], y[6], y[7], r[8]],
            [r[0], r[1], w[2], r[3], r[4], w[5], r[6], r[7], w[8]],
            [y[8], o[1], o[2], y[5], o[4], o[5], y[2], o[7], o[8]],
            g,
            [b[2], b[5], b[8], b[1], b[4], b[7], b[0], b[3], b[6]]]
        self._hash = None

    def validate(self):
        from collections import defaultdict
        from func import flatten
        _map = defaultdict(lambda: 0)
        for g in flatten(self.data):
            _map[g] += 1
        return max(_map.values()) == 9
