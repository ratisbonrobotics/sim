import numpy as np
import xml.etree.ElementTree as ET
import mujoco
from mujoco import mjx
import mediapy as media
import jax
import jax.numpy as jp
from functools import partial
import time

# Obstacle locations
OBSTACLE_LOCS = np.array([[0, 2], [0.5, 2]])

# Simulation parameters
DURATION = 2  # seconds
FRAMERATE = 30  # fps
NUM_ENVS = 8000  # Batch size for depth map calculation

# Camera parameters
DCAM_PARAMS = {'fov_y': 45, 'p_x': 100, 'p_y': 100}

# --- Utility Functions ---
def normalize(v):
    """Normalize a vector."""
    return v / jp.linalg.norm(v)
b_normalize = jax.vmap(jax.vmap(normalize))

def get_ray_vectors_cf(dcam_pars):
    """Generate ray vectors in camera frame."""
    fov_y, p_x, p_y = dcam_pars['fov_y'], dcam_pars['p_x'], dcam_pars['p_y']
    _fov_y = np.deg2rad(fov_y)
    f = 0.1  # Fixed focal length
    h_ip = np.tan(_fov_y / 2) * 2 * f  # Image plane height
    w_ip = h_ip * (p_x / p_y)  # Image plane width (square pixels)
    delta = w_ip / (2 * p_x)
    x_coords_ip = np.linspace(-w_ip / 2 + delta, w_ip / 2 - delta, p_x)
    y_coords_ip = np.flip(np.linspace(-h_ip / 2 + delta, h_ip / 2 - delta, p_y))
    xx, yy = np.meshgrid(x_coords_ip, y_coords_ip)
    vecs_cf = np.concatenate([
        np.expand_dims(xx, axis=2),
        np.expand_dims(yy, axis=2),
        -1 * np.ones(xx.shape + (1,)) * f
    ], axis=2)
    return vecs_cf

def cf2gf(vecs_cf, mjx_data, cam_ind):
    """Transform vectors from camera frame to global frame."""
    cam_bases = mjx_data.cam_xmat[cam_ind]
    vecs_gf = np.matmul(cam_bases, np.expand_dims(vecs_cf, 3))
    vecs_gf = np.squeeze(vecs_gf)
    return b_normalize(vecs_gf)

# --- MuJoCo Setup ---
# Load base MuJoCo XML
xml_2dof = """
<mujoco model="2 dof ball">
  <compiler angle="radian"/>
  <option timestep="0.002" iterations="1" ls_iterations="4" solver="Newton" gravity="0 0 -9.81">
    <flag eulerdamp="disable"/>
  </option>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 .125" type="plane" material="MatPlane" group="1"/>
    <body name="ball" pos="0 0 0.2">
      <camera name="fwd" pos="0 0.11 0" xyaxes="1 0 0 0 0 1"/>
      <joint axis="1 0 0" pos="0 0 -0.1" limited="false" name="rootx" type="slide"/>
      <joint axis="0 1 0" pos="0 0 -0.1" limited="false" name="rooty" type="slide"/>
      <joint axis="0 0 1" pos="0 0 -0.1" limited="false" name="rotz" type="hinge"/>
      <geom name="ball_geom" size="0.1" type="sphere" conaffinity="0" contype="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="50.0" joint="rootx"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="50.0" joint="rooty"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="50.0" joint="rotz"/>
  </actuator>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0" width="100" height="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture type="2d" name="texplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="100" height="100"/>
    <material name="MatPlane" texture="texplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
</mujoco>
"""

# Programmatically add obstacles to XML
root = ET.fromstring(xml_2dof)
e_body = root.find("worldbody")
for i, cl in enumerate(OBSTACLE_LOCS):
    xml_obst = f'<geom name="obst_{i}" size="0.1" type="capsule" fromto="{cl[0]} {cl[1]} 0 {cl[0]} {cl[1]} 1" conaffinity="1" contype="0"/>'
    e_body.append(ET.fromstring(xml_obst))
xml_2dof = ET.tostring(root)

# Load MuJoCo model
model = mujoco.MjModel.from_xml_string(xml_2dof)
renderer = mujoco.Renderer(model)

# Initialize MuJoCo data
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Convert to JAX arrays
mx = mjx.put_model(model)
dx = mjx.put_data(model, data)
dx = mjx.forward(mx, dx)

# --- Raycasting Setup ---
mask = None  # Collide with all geoms
f_ray = jax.jit(partial(mjx.ray, geomgroup=mask))
in_ax = (None,) * 3 + (0,)
f_mray = jax.vmap(jax.vmap(f_ray, in_axes=in_ax), in_axes=in_ax)

# Pre-compute ray vectors
vecs_cf = get_ray_vectors_cf(DCAM_PARAMS)

# Batched depth map calculation function
@jax.jit
def f_bdmap(mx, dx, vecs_gf, _):
    return f_mray(mx, dx, dx.cam_xpos[0], vecs_gf)

# --- Simulation Loop ---
frames, depth_frames = [], []
mujoco.mj_resetData(model, data)
ctrl = np.array([0, 0.1, -0.0001])

t_drender = 0
while data.time < DURATION:
    data.ctrl = ctrl
    mujoco.mj_step(model, data)

    # Capture frames and depth maps
    if len(frames) < data.time * FRAMERATE:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, "fwd")
        frames.append(renderer.render())

        dx = mjx.put_data(model, data)
        vecs_gf = cf2gf(vecs_cf, dx, 0)

        t0 = time.time()
        dist, _ = f_bdmap(mx, dx, vecs_gf, jp.zeros(NUM_ENVS))
        dist.block_until_ready()
        t_drender += time.time() - t0

        depth_frames.append(dist[0])

# Save videos
media.write_video("depth/vid1.gif", frames, fps=FRAMERATE, codec='gif')
media.write_video("depth/vid2.gif", depth_frames, fps=FRAMERATE, codec='gif')

print("Depth rendering at {} fps".format(len(depth_frames) * NUM_ENVS / t_drender))

renderer.close()