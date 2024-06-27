import jax
import jax.numpy as jnp
from jax import jit, vmap
from PIL import Image
import numpy as np
import moviepy.editor as mpy

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 30
DURATION = 5  # seconds
NUM_FRAMES = FPS * DURATION

def parse_obj_file(file_path):
    vertices, texture_coords, faces = [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '): vertices.append([float(coord) for coord in line.split()[1:]])
            elif line.startswith('vt '): texture_coords.append([float(coord) for coord in line.split()[1:]])
            elif line.startswith('f '):
                face = []
                for vertex_index in line.split()[1:]:
                    indices = vertex_index.split('/')
                    face.append((int(indices[0]) - 1, int(indices[1]) - 1 if len(indices) > 1 else None))
                faces.append(face)
    return jnp.array(vertices), jnp.array(texture_coords), jnp.array(faces)

@jit
def create_projection_matrix(fov, aspect_ratio, near, far):
    f = 1 / jnp.tan(fov / 2)
    return jnp.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])

@jit
def create_view_matrix(eye, center, up):
    f = (center - eye) / jnp.linalg.norm(center - eye)
    s = jnp.cross(f, up) / jnp.linalg.norm(jnp.cross(f, up))
    u = jnp.cross(s, f)
    
    return jnp.array([
        [s[0], s[1], s[2], -jnp.dot(s, eye)],
        [u[0], u[1], u[2], -jnp.dot(u, eye)],
        [-f[0], -f[1], -f[2], jnp.dot(f, eye)],
        [0, 0, 0, 1]
    ])

@jit
def create_model_matrix(scale, rotation, translation):
    scale_matrix = jnp.diag(jnp.concatenate([scale, jnp.array([1])]))
    
    rx, ry, rz = rotation
    rotation_x = jnp.array([
        [1, 0, 0, 0],
        [0, jnp.cos(rx), -jnp.sin(rx), 0],
        [0, jnp.sin(rx), jnp.cos(rx), 0],
        [0, 0, 0, 1]
    ])
    rotation_y = jnp.array([
        [jnp.cos(ry), 0, jnp.sin(ry), 0],
        [0, 1, 0, 0],
        [-jnp.sin(ry), 0, jnp.cos(ry), 0],
        [0, 0, 0, 1]
    ])
    rotation_z = jnp.array([
        [jnp.cos(rz), -jnp.sin(rz), 0, 0],
        [jnp.sin(rz), jnp.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    rotation_matrix = rotation_z @ rotation_y @ rotation_x
    
    translation_matrix = jnp.eye(4).at[:3, 3].set(translation)
    
    return translation_matrix @ rotation_matrix @ scale_matrix

@jit
def rasterize_triangle(vertices, texture_coords, face):
    v0, v1, v2 = [vertices[i] for i, _ in face]
    vt0, vt1, vt2 = [texture_coords[i] for _, i in face if i is not None]

    v0, v1, v2 = v0[:3] / v0[3], v1[:3] / v1[3], v2[:3] / v2[3]

    v0 = ((v0[0] + 1) * WIDTH / 2, (1 - v0[1]) * HEIGHT / 2, v0[2])
    v1 = ((v1[0] + 1) * WIDTH / 2, (1 - v1[1]) * HEIGHT / 2, v1[2])
    v2 = ((v2[0] + 1) * WIDTH / 2, (1 - v2[1]) * HEIGHT / 2, v2[2])

    def edge_function(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    x, y = jnp.meshgrid(jnp.arange(WIDTH), jnp.arange(HEIGHT))
    points = jnp.stack([x, y], axis=-1)

    area = edge_function(v0, v1, v2)
    w0 = vmap(vmap(lambda p: edge_function(v1, v2, p)))(points) / area
    w1 = vmap(vmap(lambda p: edge_function(v2, v0, p)))(points) / area
    w2 = 1 - w0 - w1

    mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
    tx = w0 * vt0[0] + w1 * vt1[0] + w2 * vt2[0]
    ty = 1 - (w0 * vt0[1] + w1 * vt1[1] + w2 * vt2[1])

    return mask, depth, tx, ty

@jit
def create_rotation_matrix(angle, axis):
    c, s = jnp.cos(angle), jnp.sin(angle)
    
    x_axis = jnp.array([[1, 0, 0, 0],
                        [0, c, -s, 0],
                        [0, s, c, 0],
                        [0, 0, 0, 1]])
    
    y_axis = jnp.array([[c, 0, s, 0],
                        [0, 1, 0, 0],
                        [-s, 0, c, 0],
                        [0, 0, 0, 1]])
    
    z_axis = jnp.array([[c, -s, 0, 0],
                        [s, c, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    
    return jax.lax.switch(axis, [lambda: x_axis, lambda: y_axis, lambda: z_axis])

@jit
def create_model_matrices(t):
    rotation_angle = t * 2 * jnp.pi / DURATION
    return jnp.array([
        [create_model_matrix(jnp.array([0.1, 0.1, 0.1]), jnp.array([0, rotation_angle, 0]), jnp.array([-1.5, 0, -3])) @ create_rotation_matrix(rotation_angle, 1),
         create_model_matrix(jnp.array([1.0, 1.0, 1.0]), jnp.array([0, rotation_angle * 0.5, 0]), jnp.array([0.2, 0, -4])) @ create_rotation_matrix(rotation_angle * 0.5, 1)],
        [create_model_matrix(jnp.array([0.1, 0.1, 0.1]), jnp.array([0, rotation_angle * 1.5, 0]), jnp.array([-0.4, 0, -3])) @ create_rotation_matrix(rotation_angle * 1.5, 1),
         create_model_matrix(jnp.array([1.0, 1.0, 1.0]), jnp.array([0, rotation_angle * 0.75, 0]), jnp.array([0.2, 0, -4])) @ create_rotation_matrix(rotation_angle * 0.75, 1)]
    ])

@jit
def render_scene(vertices, texture_coords, faces, texture, mvp_matrix):
    depth_buffer = jnp.full((HEIGHT, WIDTH), jnp.inf)
    color_buffer = jnp.zeros((HEIGHT, WIDTH, 3), dtype=jnp.uint8)

    vertices_homogeneous = jnp.pad(vertices, ((0, 0), (0, 1)), constant_values=1)
    vertices_transformed = jnp.dot(vertices_homogeneous, mvp_matrix.T)

    def render_face(buffers, face):
        depth_buffer, color_buffer = buffers
        mask, depth, tx, ty = rasterize_triangle(vertices_transformed, texture_coords, face)
        
        update = mask & (depth < depth_buffer)
        depth_buffer = jnp.where(update, depth, depth_buffer)
        
        tx_clipped = jnp.clip(tx * texture.shape[1], 0, texture.shape[1] - 1).astype(jnp.int32)
        ty_clipped = jnp.clip(ty * texture.shape[0], 0, texture.shape[0] - 1).astype(jnp.int32)
        color = texture[ty_clipped, tx_clipped]
        
        color_buffer = jnp.where(update[:, :, jnp.newaxis], color, color_buffer)
        
        return (depth_buffer, color_buffer), None

    (depth_buffer, color_buffer), _ = jax.lax.scan(render_face, (depth_buffer, color_buffer), faces)
    
    return depth_buffer, color_buffer

@jit
def create_frame(vertices1, texture_coords1, faces1, texture1, 
                 vertices2, texture_coords2, faces2, texture2, 
                 view_projection_matrix, model_matrix1, model_matrix2):
    mvp_matrix1 = view_projection_matrix @ model_matrix1
    mvp_matrix2 = view_projection_matrix @ model_matrix2
    
    depth_buffer1, color_buffer1 = render_scene(vertices1, texture_coords1, faces1, texture1, mvp_matrix1)
    depth_buffer2, color_buffer2 = render_scene(vertices2, texture_coords2, faces2, texture2, mvp_matrix2)
    
    # Combine the two renders with proper depth testing
    mask = depth_buffer1 <= depth_buffer2
    combined_color = jnp.where(mask[:, :, jnp.newaxis], color_buffer1, color_buffer2)
    
    return combined_color

vectorized_create_frame = jit(vmap(create_frame, in_axes=(None, None, None, None,
                                                          None, None, None, None,
                                                          None, 0, 0)))

@jit
def create_model_matrices(t):
    rotation_angle = t * 2 * jnp.pi / DURATION
    return jnp.array([
        [create_model_matrix(jnp.array([0.1, 0.1, 0.1]), jnp.array([0, rotation_angle, 0]), jnp.array([-1.5, 0, -3])) @ create_rotation_matrix(rotation_angle, 1),
         create_model_matrix(jnp.array([1.0, 1.0, 1.0]), jnp.array([0, rotation_angle * 0.5, 0]), jnp.array([0.2, 0, -4])) @ create_rotation_matrix(rotation_angle * 0.5, 1)],
        [create_model_matrix(jnp.array([0.1, 0.1, 0.1]), jnp.array([0, rotation_angle * 1.5, 0]), jnp.array([-0.4, 0, -3])) @ create_rotation_matrix(rotation_angle * 1.5, 1),
         create_model_matrix(jnp.array([1.0, 1.0, 1.0]), jnp.array([0, rotation_angle * 0.75, 0]), jnp.array([0.2, 0, -4])) @ create_rotation_matrix(rotation_angle * 0.75, 1)]
    ])

def main():
    vertices1, texture_coords1, faces1 = parse_obj_file('drone.obj')
    texture1 = jnp.array(Image.open('drone.png').convert('RGB'))

    vertices2, texture_coords2, faces2 = parse_obj_file('african_head.obj')
    texture2 = jnp.array(Image.open('african_head_diffuse.tga').convert('RGB'))
    
    aspect_ratio = WIDTH / HEIGHT
    fov = jnp.radians(45)
    near, far = 0.1, 100.0
    
    view_matrix = create_view_matrix(jnp.array([0, 0, 3]), jnp.array([0, 0, 0]), jnp.array([0, 1, 0]))
    projection_matrix = create_projection_matrix(fov, aspect_ratio, near, far)
    view_projection_matrix = projection_matrix @ view_matrix
    
    frames = []
    for t in jnp.linspace(0, DURATION, NUM_FRAMES):
        model_matrices = create_model_matrices(t)
        frames.append(vectorized_create_frame(vertices1, texture_coords1, faces1, texture1,
                                              vertices2, texture_coords2, faces2, texture2,
                                              view_projection_matrix, 
                                              model_matrices[:, 0], model_matrices[:, 1]))
    
    all_frames_np = np.array(frames)
    
    for i in range(2):
        clip = mpy.ImageSequenceClip([frame[i] for frame in all_frames_np], fps=FPS)
        clip.write_videofile(f"output_{i+1}.mp4")

if __name__ == '__main__':
    main()