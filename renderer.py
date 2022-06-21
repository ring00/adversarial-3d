import os
import ModernGL
from objloader import Obj
from PIL import Image
from pyrr import Matrix44, Vector3
import numpy as np


class Renderer(object):

    def __init__(self, viewport=(299, 299)):
        """
        Construct a Renderer object

        viewport: width and height of windows
        """
        super(Renderer, self).__init__()
        self.width, self.height = viewport

        # require OpenGL 330 core profile
        self.ctx = ModernGL.create_standalone_context(viewport, require=330)
        self.ctx.enable(ModernGL.DEPTH_TEST)
        self.ctx.enable(ModernGL.CULL_FACE)

        # frame buffer object
        self.fbo = self.ctx.framebuffer(
            self.ctx.texture(viewport, components=2, floats=True),
            self.ctx.depth_renderbuffer(viewport)
        )

        # shader program
        self.prog = self.ctx.program([
            self.ctx.vertex_shader('''
                #version 330 core

                // model view projection matrix
                uniform mat4 mvp;

                in vec3 in_vert;
                in vec2 in_text;

                out vec2 v_text;

                void main() {
                    gl_Position = mvp * vec4(in_vert, 1.0f);
                    // UV coordinate starts from bottom left corner
                    // image coordinate starts from top left corner
                    // therefore we need to flip the V-axis
                    v_text = vec2(in_text.x, 1.0f - in_text.y);
                }
            '''),
            self.ctx.fragment_shader('''
                #version 330 core

                in vec2 v_text;

                out vec2 f_color;

                void main() {
                    f_color = v_text;
                }
            ''')
        ])

        self.mvp = self.prog.uniforms['mvp']

    def load_obj(self, filename):
        if not os.path.isfile(filename):
            print('{} is not an existing regular file!'.format(filename))
            return

        obj = Obj.open(filename)

        # TODO: not very efficient, consider using an element index array later
        self.vao = self.ctx.simple_vertex_array(
            self.prog,
            self.ctx.buffer(obj.pack('vx vy vz tx ty')),
            ['in_vert', 'in_text']
        )

    def set_parameters(self,
                       camera_distance=(2.5, 3.0),
                       x_translation=(-0.05, 0.05),
                       y_translation=(-0.05, 0.05),
                       deflection=1.0):
        """
        Set parameters for rendering.

        camera_distance: the minimum and maximum distance from camera
        x_translation: the minimum and maximum translation along x-axis
        y_translation: the minimum and maximum translation along y-axis
        deflection: the magnitude of the rotation, see rand_rotation_matrix
        """
        self.close, self.far = camera_distance
        self.x_low, self.x_high = x_translation
        self.y_low, self.y_high = y_translation
        self.deflection = deflection

    @staticmethod
    def rand_rotation_matrix(deflection=1.0, randnums=None):
        """
        Creates a random rotation matrix.

        deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
        rotation. Small deflection => small perturbation.
        randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
        """
        if randnums is None:
            randnums = np.random.uniform(size=(3,))

        theta, phi, z = randnums

        # Rotation about the pole (Z).
        theta = theta * 2.0 * deflection * np.pi
        phi = phi * 2.0 * np.pi  # For direction of pole deflection.
        z = z * 2.0 * deflection  # For magnitude of pole deflection.

        # Compute a vector V used for distributing points over the sphere
        # via the reflection I - V Transpose(V).  This formulation of V
        # will guarantee that if x[1] and x[2] are uniformly distributed,
        # the reflected points will be uniform on the sphere.  Note that V
        # has length sqrt(2) to eliminate the 2 in the Householder matrix.

        r = np.sqrt(z)
        V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z)
        )
        Vx, Vy, Vz = V

        st = np.sin(theta)
        ct = np.cos(theta)

        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

        # Construct the rotation matrix  ( V Transpose(V) - I ) R.
        M = (np.outer(V, V) - np.eye(3)).dot(R)

        return M

    def render(self, batch_size):
        warp = np.empty(
            (batch_size, self.height, self.width, 2), dtype=np.float32)
        for i in range(batch_size):
            translation = Matrix44.from_translation((
                np.random.uniform(self.x_low, self.x_high),
                np.random.uniform(self.y_low, self.y_high),
                0.0
            ))
            rotation = Matrix44.from_matrix33(
                self.rand_rotation_matrix(self.deflection)
            )
            view = Matrix44.look_at(
                (0.0, 0.0, np.random.uniform(self.close, self.far)),
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            )
            projection = Matrix44.perspective_projection(
                45.0, self.width / self.height, 0.1, 1000.0
            )

            # TODO: translation or rotation first?
            transform = projection * view * translation * rotation

            self.fbo.use()
            self.fbo.clear()

            self.mvp.write(transform.astype('f4').tobytes())
            self.vao.render()

            framebuffer = self.fbo.read(components=2, floats=True)
            warp[i] = np.frombuffer(framebuffer, dtype=np.float32).reshape(
                (self.height, self.width, 2))[::-1]

        return warp
