import mujoco
from mujoco import mjx
import jax
import mediapy as media

xml = """
<mujoco model="CF2 scene">

  <option integrator="RK4" density="1.225" viscosity="1.8e-5"/>

  <compiler inertiafromgeom="false" meshdir="assets" autolimits="true"/>

  <default>
    <default class="cf2">
      <default class="visual">
        <geom group="2" type="mesh" contype="0" conaffinity="0"/>
      </default>
      <default class="collision">
        <geom group="3" type="mesh"/>
      </default>
      <site group="5"/>
    </default>
  </default>

  <asset>
    <material name="polished_plastic" rgba="0.631 0.659 0.678 1"/>
    <material name="polished_gold" rgba="0.969 0.878 0.6 1"/>
    <material name="medium_gloss_plastic" rgba="0.109 0.184 0.0 1"/>
    <material name="propeller_plastic" rgba="0.792 0.820 0.933 1"/>
    <material name="white" rgba="1 1 1 1"/>
    <material name="body_frame_plastic" rgba="0.102 0.102 0.102 1"/>
    <material name="burnished_chrome" rgba="0.898 0.898 0.898 1"/>

    <mesh file="cf2_0.obj"/>
    <mesh file="cf2_1.obj"/>
    <mesh file="cf2_2.obj"/>
    <mesh file="cf2_3.obj"/>
    <mesh file="cf2_4.obj"/>
    <mesh file="cf2_5.obj"/>
    <mesh file="cf2_6.obj"/>
    <mesh file="cf2_collision_0.obj"/>
    <mesh file="cf2_collision_1.obj"/>
    <mesh file="cf2_collision_2.obj"/>
    <mesh file="cf2_collision_3.obj"/>
    <mesh file="cf2_collision_4.obj"/>
    <mesh file="cf2_collision_5.obj"/>
    <mesh file="cf2_collision_6.obj"/>
    <mesh file="cf2_collision_7.obj"/>
    <mesh file="cf2_collision_8.obj"/>
    <mesh file="cf2_collision_9.obj"/>
    <mesh file="cf2_collision_10.obj"/>
    <mesh file="cf2_collision_11.obj"/>
    <mesh file="cf2_collision_12.obj"/>
    <mesh file="cf2_collision_13.obj"/>
    <mesh file="cf2_collision_14.obj"/>
    <mesh file="cf2_collision_15.obj"/>
    <mesh file="cf2_collision_16.obj"/>
    <mesh file="cf2_collision_17.obj"/>
    <mesh file="cf2_collision_18.obj"/>
    <mesh file="cf2_collision_19.obj"/>
    <mesh file="cf2_collision_20.obj"/>
    <mesh file="cf2_collision_21.obj"/>
    <mesh file="cf2_collision_22.obj"/>
    <mesh file="cf2_collision_23.obj"/>
    <mesh file="cf2_collision_24.obj"/>
    <mesh file="cf2_collision_25.obj"/>
    <mesh file="cf2_collision_26.obj"/>
    <mesh file="cf2_collision_27.obj"/>
    <mesh file="cf2_collision_28.obj"/>
    <mesh file="cf2_collision_29.obj"/>
    <mesh file="cf2_collision_30.obj"/>
    <mesh file="cf2_collision_31.obj"/>
  </asset>

  <worldbody>
    <body name="cf2" pos="0 0 0.1" childclass="cf2">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.027" diaginertia="2.3951e-5 2.3951e-5 3.2347e-5"/>
      <camera name="track" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <geom mesh="cf2_0" material="propeller_plastic" class="visual"/>
      <geom mesh="cf2_1" material="medium_gloss_plastic" class="visual"/>
      <geom mesh="cf2_2" material="polished_gold" class="visual"/>
      <geom mesh="cf2_3" material="polished_plastic" class="visual"/>
      <geom mesh="cf2_4" material="burnished_chrome" class="visual"/>
      <geom mesh="cf2_5" material="body_frame_plastic" class="visual"/>
      <geom mesh="cf2_6" material="white" class="visual"/>
      <geom mesh="cf2_collision_0" class="collision"/>
      <geom mesh="cf2_collision_1" class="collision"/>
      <geom mesh="cf2_collision_2" class="collision"/>
      <geom mesh="cf2_collision_3" class="collision"/>
      <geom mesh="cf2_collision_4" class="collision"/>
      <geom mesh="cf2_collision_5" class="collision"/>
      <geom mesh="cf2_collision_6" class="collision"/>
      <geom mesh="cf2_collision_7" class="collision"/>
      <geom mesh="cf2_collision_8" class="collision"/>
      <geom mesh="cf2_collision_9" class="collision"/>
      <geom mesh="cf2_collision_10" class="collision"/>
      <geom mesh="cf2_collision_11" class="collision"/>
      <geom mesh="cf2_collision_12" class="collision"/>
      <geom mesh="cf2_collision_13" class="collision"/>
      <geom mesh="cf2_collision_14" class="collision"/>
      <geom mesh="cf2_collision_15" class="collision"/>
      <geom mesh="cf2_collision_16" class="collision"/>
      <geom mesh="cf2_collision_17" class="collision"/>
      <geom mesh="cf2_collision_18" class="collision"/>
      <geom mesh="cf2_collision_19" class="collision"/>
      <geom mesh="cf2_collision_20" class="collision"/>
      <geom mesh="cf2_collision_21" class="collision"/>
      <geom mesh="cf2_collision_22" class="collision"/>
      <geom mesh="cf2_collision_23" class="collision"/>
      <geom mesh="cf2_collision_24" class="collision"/>
      <geom mesh="cf2_collision_25" class="collision"/>
      <geom mesh="cf2_collision_26" class="collision"/>
      <geom mesh="cf2_collision_27" class="collision"/>
      <geom mesh="cf2_collision_28" class="collision"/>
      <geom mesh="cf2_collision_29" class="collision"/>
      <geom mesh="cf2_collision_30" class="collision"/>
      <geom mesh="cf2_collision_31" class="collision"/>
      <site name="imu"/>
      <site name="actuation"/>
    </body>
  </worldbody>

  <actuator>
    <motor class="cf2" ctrlrange="0 0.35" gear="0 0 1 0 0 0" site="actuation" name="body_thrust"/>
    <motor class="cf2" ctrlrange="-1 1" gear="0 0 0 -0.00001 0 0" site="actuation" name="x_moment"/>
    <motor class="cf2" ctrlrange="-1 1" gear="0 0 0 0 -0.00001 0" site="actuation" name="y_moment"/>
    <motor class="cf2" ctrlrange="-1 1" gear="0 0 0 0 0 -0.00001" site="actuation" name="z_moment"/>
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
    <global azimuth="-20" elevation="-20" ellipsoidinertia="true"/>
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

# Make model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

mjx_model = mjx.put_model(model)

# Make renderer, render and show the pixels
renderer = mujoco.Renderer(model)

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

jit_step = jax.jit(mjx.step)

# Simulate and display video.
frames = []
mujoco.mj_resetData(model, data)  # Reset state and time.
mjx_data = mjx.put_data(model, data)
while mjx_data.time < duration:
    mjx_data = jit_step(mjx_model, mjx_data)
    if len(frames) < mjx_data.time * framerate:
        mj_data = mjx.get_data(model, mjx_data)
        renderer.update_scene(mj_data)
        pixels = renderer.render()
        frames.append(pixels)

renderer.close()

# Save video from frames
media.write_video('simulation_mjx.mp4', frames, fps=framerate)