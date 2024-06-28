import jax
import jax.numpy as jnp
from jax import jit, vmap
from PIL import Image
import numpy as np
from moviepy.editor import ImageSequenceClip
import os

# Constants
WIDTH, HEIGHT = 640, 480

def parse_obj_file(file_path):
    vertices, texture_coords, faces, materials = [], [], [], []
    current_material = None
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '): vertices.append([float(coord) for coord in line.split()[1:]])
            elif line.startswith('vt '): texture_coords.append([float(coord) for coord in line.split()[1:]])
            elif line.startswith('f '):
                face = []
                for vertex_index in line.split()[1:]:
                    indices = vertex_index.split('/')
                    face.append((int(indices[0]) - 1, int(indices[1]) - 1 if len(indices) > 1 and indices[1] else None))
                faces.append(face)
                materials.append(current_material)
            elif line.startswith('usemtl '): current_material = line.split()[1]
    return jnp.array(vertices), jnp.array(texture_coords), jnp.array(faces), materials

def parse_mtl_file(file_path):
    materials = {}
    current_material = None
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('newmtl '):
                current_material = line.split()[1]
                materials[current_material] = {'Kd': None, 'map_Kd': None}
            elif line.startswith('Kd '):
                materials[current_material]['Kd'] = [float(v) for v in line.split()[1:]]
            elif line.startswith('map_Kd '):
                materials[current_material]['map_Kd'] = line.split()[1]
    return materials

def create_projection_matrix(fov, aspect_ratio, near, far):
    f = 1 / jnp.tan(fov / 2)
    return jnp.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])

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

def create_model_matrix(scale, rotation, translation):
    scale = jnp.array(scale)
    rotation = jnp.array(rotation)
    translation = jnp.array(translation)
    
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
    
    translation_matrix = jnp.eye(4)
    translation_matrix = translation_matrix.at[:3, 3].set(translation)
    
    return translation_matrix @ rotation_matrix @ scale_matrix

@jit
def rasterize_triangle(vertices, texture_coords, face, color):
    v0, v1, v2 = [vertices[i] for i, _ in face]
    vt0, vt1, vt2 = [texture_coords[i] for _, i in face if i is not None]

    # Perspective division
    v0 = v0[:3] / v0[3]
    v1 = v1[:3] / v1[3]
    v2 = v2[:3] / v2[3]

    # Convert to screen space
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

    return mask, depth, tx, ty, color

@jit
def render_scene(vertices, texture_coords, faces, materials, textures, mvp_matrix):
    depth_buffer = jnp.full((HEIGHT, WIDTH), jnp.inf)
    color_buffer = jnp.zeros((HEIGHT, WIDTH, 3), dtype=jnp.uint8)

    # Apply MVP matrix to vertices
    vertices_homogeneous = jnp.pad(vertices, ((0, 0), (0, 1)), constant_values=1)
    vertices_transformed = jnp.dot(vertices_homogeneous, mvp_matrix.T)

    @jit
    def render_face(buffers, face_data):
        depth_buffer, color_buffer = buffers
        face, material = face_data
        
        if material['map_Kd'] is not None:
            texture = textures[material['map_Kd']]
            color = None
        else:
            texture = None
            color = jnp.array(material['Kd']) * 255
        
        mask, depth, tx, ty, face_color = rasterize_triangle(vertices_transformed, texture_coords, face, color)
        
        update = mask & (depth < depth_buffer)
        depth_buffer = jnp.where(update, depth, depth_buffer)
        
        if texture is not None:
            tx_clipped = jnp.clip(tx * texture.shape[1], 0, texture.shape[1] - 1).astype(jnp.int32)
            ty_clipped = jnp.clip(ty * texture.shape[0], 0, texture.shape[0] - 1).astype(jnp.int32)
            color = texture[ty_clipped, tx_clipped]
        else:
            color = face_color
        
        color_buffer = jnp.where(update[:, :, jnp.newaxis], color, color_buffer)
        
        return (depth_buffer, color_buffer), None

    (depth_buffer, color_buffer), _ = jax.lax.scan(render_face, (depth_buffer, color_buffer), (faces, materials))
    
    return color_buffer

@jit
def create_model_matrices(frame, num_frames):
    t = frame / num_frames * 2 * jnp.pi  # Time parameter
    return jnp.array([
        create_model_matrix(scale=[0.1, 0.1, 0.1], rotation=[0, t, 0], translation=[0, 0, -3]),
        create_model_matrix(scale=[0.1, 0.1, 0.1], rotation=[0, 2*t, 0], translation=[0.5, 0, -4])
    ])

# Define batched_render_scene at the global scope
batched_render_scene = jit(vmap(render_scene, in_axes=(None, None, None, None, None, 0)))

@jit
def render_frame(carry, frame):
    projection_matrix, view_matrix, vertices, texture_coords, faces, materials, textures, num_frames = carry
    
    model_matrices = create_model_matrices(frame, num_frames)
    mvp_matrices = jnp.einsum('ij,kjl->kil', projection_matrix @ view_matrix, model_matrices)
    
    images = batched_render_scene(
        vertices,
        texture_coords,
        faces,
        materials,
        textures,
        mvp_matrices
    )
    
    return carry, images

import time

def main():
    # Load complex scene
    vertices, texture_coords, faces, material_names = parse_obj_file('scene0/scene.obj')
    mtl_data = parse_mtl_file('scene0/scene.mtl')
    
    # Load textures
    textures = {}
    for material in mtl_data.values():
        if material['map_Kd']:
            texture_path = os.path.join('scene0', material['map_Kd'])
            textures[material['map_Kd']] = jnp.array(Image.open(texture_path).convert('RGB'))
    
    # Prepare materials for rendering
    materials = [mtl_data[name] for name in material_names]
    
    aspect_ratio = WIDTH / HEIGHT
    fov = jnp.radians(45)
    near = 0.1
    far = 100.0
    
    view_matrix = create_view_matrix(eye=jnp.array([0, 0, 3]), center=jnp.array([0, 0, 0]), up=jnp.array([0, 1, 0]))
    projection_matrix = create_projection_matrix(fov, aspect_ratio, near, far)
    
    # Animation parameters
    num_frames = 120
    fps = 30
    
    # Prepare inputs for jax.lax.scan
    carry = (projection_matrix, view_matrix, 
             vertices,
             texture_coords,
             faces,
             materials,
             textures,
             num_frames)
    
    # Compile the function
    compiled_render = jit(lambda carry, x: jax.lax.scan(render_frame, carry, x))
    
    # Warm-up run
    _ = compiled_render(carry, jnp.arange(1))
    
    # Benchmark
    start_time = time.time()
    _, all_images = compiled_render(carry, jnp.arange(num_frames))
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_frame = total_time / num_frames
    
    print(f"Total rendering time: {total_time:.2f} seconds")
    print(f"Average time per frame: {avg_time_per_frame:.4f} seconds")
    print(f"Frames per second: {1/avg_time_per_frame:.2f}")
    
    # Convert to numpy arrays
    frames = np.array(all_images)
    
    # Create and save the animation
    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_videofile("output_complex_scene.mp4")

if __name__ == '__main__':
    main()