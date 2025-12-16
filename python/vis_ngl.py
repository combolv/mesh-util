import moderngl
import moderngl_window as mglw
import numpy as np
from matplotlib import cm
from backend import ComputeTetXYIntersectionGpu

tet_verts = np.load("vert_tetra.npy")
tet_indices = np.load("elems_tetra.npy")
num_tets = tet_indices.shape[0]
# Check the minimum and maximum coordinates.
min_z = np.min(tet_verts[:, 2])
max_z = np.max(tet_verts[:, 2])
center_z = 0.5 * (min_z + max_z)

tet_verts[:, 2] -= 0.8 # Center the mesh at z = 0.
vert_field = np.load("sol2.npy")  # Load the scalar field at vertices.
# Normalize the vert_field to [0, 1]

xy_loc, verts = ComputeTetXYIntersectionGpu(tet_verts.T, tet_indices.T, vert_field)

class TriangleRenderer(mglw.WindowConfig):
    window_size = (800, 800)
    title = "Triangle Visualization (Custom Data)"
    resource_dir = '.'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # -----------------------------
        # (1) LOAD YOUR DATA HERE
        # -----------------------------
        # your numpy arrays:
        # pos: shape [2, 3K]
        # scalars: shape [3K]

        # EXAMPLE (replace with your real data)
        K = 500
        pos = xy_loc # np.random.randn(2, 3*K).astype(np.float32)
        scalars = verts # np.random.rand(3*K).astype(np.float32)

        # -----------------------------
        # (2) NORMALIZE POSITIONS
        # -----------------------------
        min_xy = pos.min(axis=1, keepdims=True)
        max_xy = pos.max(axis=1, keepdims=True)
        centered = (pos - (min_xy + max_xy) / 2.0)
        scale = (max_xy - min_xy).max()
        pos_norm = centered / (scale / 1.8)  # fit inside [-1, 1]

        # Flatten to [N, 2]
        pos_flat = pos_norm.T.reshape(-1, 2)  # (3K, 2)

        # -----------------------------
        # (3) NORMALIZE SCALARS TO [0, 1]
        # -----------------------------
        smin = scalars.min()
        smax = scalars.max()
        scalars_norm = (scalars - smin) / (smax - smin + 1e-12)

        # -----------------------------
        # (4) CREATE COLORS USING A COLORMAP
        # -----------------------------
        cmap = cm.get_cmap('viridis')
        colors = cmap(scalars_norm)[:, :3].astype(np.float32)  # (3K, 3)

        # concatenate: [pos.x, pos.y, r, g, b]
        vertices = np.hstack([pos_flat, colors]).astype('f4').ravel()

        # -----------------------------
        # (5) Moderngl programs
        # -----------------------------
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec2 in_pos;
                in vec3 in_color;
                out vec3 v_color;

                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330

                in vec3 v_color;
                out vec4 f_color;

                void main() {
                    f_color = vec4(v_color, 1.0);
                }
            """
        )

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.simple_vertex_array(
            self.prog, self.vbo,
            'in_pos', 'in_color'
        )

        self.num_vertices = pos_flat.shape[0]

    # IMPORTANT: override on_render
    def on_render(self, time, frame_time):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.vao.render(moderngl.TRIANGLES)


if __name__ == '__main__':
    # TriangleRenderer.run(xy_loc=xy_loc, verts=verts)

    mglw.run_window_config(TriangleRenderer)
