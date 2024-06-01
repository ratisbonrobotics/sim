import jax
import mujoco
import typing
import mediapy
from mujoco import mjx

#jax.config.update("jax_compilation_cache_dir", "/home/markusheimerl/sim/cache")

xml = """
<mujoco model="Simple Drone">
  <option integrator="RK4" density="1.225" viscosity="1.8e-5"/>

  <compiler inertiafromgeom="true" autolimits="true"/>

  <default>
    <geom type="capsule" size=".01" rgba="0.8 0.6 0.4 1"/>
    <joint type="hinge" limited="true" range="-45 45"/>
  </default>

  <worldbody>
    <body name="drone" pos="0 0 0.1">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.027" diaginertia="2.3951e-5 2.3951e-5 3.2347e-5"/>
      <camera name="track" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>

      <!-- Crossing rods -->
      <geom name="rod1" type="capsule" fromto="-0.1 0 0 0.1 0 0" size="0.005"/>
      <geom name="rod2" type="capsule" fromto="0 -0.1 0 0 0.1 0" size="0.005"/>

      <!-- Center ball -->
      <geom name="center_ball" type="sphere" pos="0 0 0" size="0.02"/>

      <!-- Rotor balls -->
      <geom name="rotor_ball1" type="sphere" pos="0.1 0.0 0" size="0.01"/>
      <geom name="rotor_ball2" type="sphere" pos="-0.1 0.0 0" size="0.01"/>
      <geom name="rotor_ball3" type="sphere" pos="0.0 -0.1 0" size="0.01"/>
      <geom name="rotor_ball4" type="sphere" pos="0.0 0.1 0" size="0.01"/>

      <site name="imu"/>
      <site name="actuation"/>
    </body>
  </worldbody>

  <actuator>
    <motor ctrlrange="0 5.35" gear="0 0 1 0 0 0" site="actuation" name="body_thrust"/>
    <motor ctrlrange="-1 1" gear="0 0 0 -0.00001 0 0" site="actuation" name="x_moment"/>
    <motor ctrlrange="-1 1" gear="0 0 0 0 -0.00001 0" site="actuation" name="y_moment"/>
    <motor ctrlrange="-1 1" gear="0 0 0 0 0 -0.00001" site="actuation" name="z_moment"/>
  </actuator>

  <sensor>
    <gyro name="body_gyro" site="imu"/>
    <accelerometer name="body_linacc" site="imu"/>
    <framequat name="body_quat" objtype="site" objname="imu"/>
  </sensor>

  <keyframe>
    <key name="hover" qpos="0 0 0.1 1 0 0 0" ctrl="0.26487 0 0 0"/>
  </keyframe>

  <statistic center="0 0 0.1" extent="0.2" meansize=".05"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-20" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""
CAMERASIZE = 128

# Make model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# Move to accelerator
mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)

# vmap mjx.ray
vray = jax.vmap(mjx.ray, (None, None, 0, 0), (0, 0))

# Define sim function
def sim(mjx_m: mjx.Model, mjx_d: mjx.Data):
  x, y = jax.numpy.meshgrid(jax.numpy.linspace(-0.5, 0.5, CAMERASIZE), jax.numpy.linspace(-0.5, 0.5, CAMERASIZE))
  origins = jax.numpy.stack([x.ravel(), y.ravel(), jax.numpy.ones(CAMERASIZE**2)], axis=1)
  directions = jax.numpy.column_stack((jax.numpy.zeros((CAMERASIZE**2,)),jax.numpy.zeros((CAMERASIZE**2,)),-jax.numpy.ones((CAMERASIZE**2,))))
  counter = 0
  end_time = 1.0

  def cond_fun(carry : typing.Tuple[int, mjx.Model, mjx.Data, jax.Array]):
      _, _, mjx_d, _ = carry
      return mjx_d.time < end_time

  def body_fun(carry : typing.Tuple[int, mjx.Model, mjx.Data, jax.Array]):
      counter, mjx_m, mjx_d, depths = carry
      mjx_d = mjx.step(mjx_m, mjx_d) # one step steps for 2 ms
      depths = depths.at[counter].set(vray(mjx_m, mjx_d, origins, directions)[0])
      counter += 1
      return counter, mjx_m, mjx_d, depths

  depths = jax.numpy.zeros((int(end_time / 0.002), CAMERASIZE**2), dtype=float)
  return jax.lax.while_loop(cond_fun, body_fun, (counter, mjx_m, mjx_d, depths))

# simulate
counter, mjx_m, mjx_d, depth = jax.jit(sim)(mjx_model, mjx_data)
depth = jax.device_get(depth)

# save the frames as MP4
min_depth = depth.min()
max_depth = depth.max()
frames = []

for i in range(depth.shape[0]):
    depth_image = depth[i].reshape((int(jax.numpy.sqrt(depth.shape[1])), int(jax.numpy.sqrt(depth.shape[1]))))
    depth_image = (depth_image - min_depth) / (max_depth - min_depth) * 255
    depth_image = depth_image.astype(jax.numpy.uint8)
    frames.append(depth_image)

mediapy.write_video("depth/vid1.mp4", frames, fps=1.0 / 0.002)