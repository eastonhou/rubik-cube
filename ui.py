import numpy as np
import cv2
from glumpy import gl, glm, gloo, app
from definitions import Cube

class Shader:
    def __init__(self):
        vertex = """
        uniform mat4   u_model;         // Model matrix
        uniform mat4   u_view;          // View matrix
        uniform mat4   u_projection;    // Projection matrix
        attribute vec2 a_texcoord;      // Vertex texture coordinates
        attribute vec3 a_position;      // Vertex position
        varying vec2   v_texcoord;      // Interpolated fragment texture coordinates (out)
        void main()
        {
            v_texcoord = a_texcoord;
            gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
        }
        """
        fragment = """
        uniform sampler2D u_texture;         // Texture 
        varying vec2      v_texcoord;        // Interpolated fragment texture coordinates (in)
        void main()
        {
            vec4 t_color = vec4(texture2D(u_texture, v_texcoord).rgb, 1.0);
            gl_FragColor = t_color;
        }
        """
        V = np.zeros((6,9,4), [("a_position", np.float32, 3),
                        ("a_texcoord",    np.float32, 2)])
        V["a_position"] = self._make_cube_positions()
        V["a_texcoord"]  = np.zeros(shape=(6,9,4,2), dtype=np.float32)
        V = V.view(gloo.VertexBuffer)
        I = np.arange(6*9*4, dtype=np.uint32)
        I = np.reshape(I, -1)
        I = I.view(gloo.IndexBuffer)
        rotation = np.eye(4, dtype=np.float32)
        program = gloo.Program(vertex, fragment)
        program.bind(V)
        program['u_model'] = rotation
        program['u_view'] = glm.translation(0, 0, -5)
        image = cv2.imread('texture.jpg', cv2.IMREAD_COLOR)
        program['u_texture'] = image[:, :, ::-1].copy().view(gloo.Texture2D)
        self.rotation = rotation
        self.program = program
        self.indices = I
        self.apply_cube(Cube())

    def apply_cube(self, cube):
        for iface in range(6):
            self.program['a_texcoord'][iface*36:(iface+1)*36] = self._apply_face(cube, iface).reshape(-1, 2)

    def rotate(self, angle, direction):
        self.rotation = glm.rotate(self.rotation, angle, *direction)

    def draw(self):
        self.program.draw(gl.GL_QUADS, self.indices)
        self.program['u_model'] = self.rotation

    def _apply_face(self, cube, iface):
        layer0 = cube.face(iface, 0)
        layer1 = cube.face(iface, 1)
        coords = np.zeros((9,4,2), dtype=np.float32)
        for i in range(9):
            x0 = layer0[i]/6
            igrid = layer1[i]
            r, c = igrid//3, igrid%3
            y, x = r/3, c/18+x0
            coords[i,0] = (x,y)
            coords[i,1] = (x+1/18,y)
            coords[i,2] = (x+1/18,y+1/3)
            coords[i,3] = (x,y+1/3)
        return coords

    def _make_cube_positions(self):
        def _make_top_face():
            face = np.zeros(shape=(9,4,3), dtype=np.float32)
            face[0] = (-1,1,-1), (-1/3,1,-1), (-1/3,1,-1/3), (-1,1,-1/3)
            for i in range(1, 9):
                r, c = i//3, i%3
                face[i] = face[0]+[(c*2/3,0,r*2/3)]
            return face
        data = np.zeros(shape=(6,9,4,3), dtype=np.float32)
        data[0] = _make_top_face()
        data[1] = -data[0].copy()
        data[1][...,0] = data[0][...,0]
        data[2] = data[0].copy()
        data[2][...,1] = -data[0][...,2]
        data[2][...,2] = 1
        data[3] = -data[2].copy()
        data[3][...,1] = data[2][...,1]
        data[4] = data[2].copy()
        data[4][...,0] = -1
        data[4][...,2] = data[2][...,0]
        data[5] = -data[4].copy()
        data[5][...,1] = data[4][...,1]
        return data

class Window:
    def __init__(self):
        config = app.configuration.get_default()
        config.samples = 16
        self.window = app.Window(height=800, width=1280, config=config)
        self.shader = Shader()
        self.cube = Cube()
        self.shader.apply_cube(self.cube)
        @self.window.event
        def on_init():
            gl.glEnable(gl.GL_DEPTH_TEST)

        @self.window.event
        def on_resize(width, height):
            self.shader.program['u_projection'] = glm.perspective(50.0, width/height, 2.0, 10.0)

        @self.window.event
        def on_mouse_drag(x, y, dx, dy, button):
            def convert(x, y):
                vs = 5 * np.tan(np.deg2rad(22.5))
                x1, y1, z1 = (x/self.window.height-0.5)*vs, -(y/self.window.height-0.5)*vs, 3
                n = np.linalg.norm((x1,y1,z1))
                return x1/n, y1/n, z1/n
            if button and (dx or dy):
                x0, y0, z0 = convert(x-dx, y-dy)
                x1, y1, z1 = convert(x, y)
                thita = y0*z1-z0*y1, -x0*z1+z0*x1, x0*y1-y0*x1
                self.shader.rotate(np.rad2deg(np.arcsin(np.linalg.norm(thita))), thita)

        @self.window.event
        def on_key_press(key, modifiers):
            if key == -1: return
            key = chr(key).upper()
            operations = 'UDFBLR'
            if key in operations:
                times = 3 if (modifiers&1) else 1
                for _ in range(times):
                    self.cube.apply_operation(key)
                self.shader.apply_cube(self.cube)

        @self.window.event
        def on_draw(dt):
            self.window.clear()
            self.shader.draw()

    def run(self):
        app.run()

if __name__ == '__main__':
    window = Window()
    window.run()