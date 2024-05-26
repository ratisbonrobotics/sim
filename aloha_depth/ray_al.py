import jax
import jax.numpy as jp
from etils import epath
import mujoco
from brax.io import mjcf
from PIL import Image

xmlGLOB = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Load the Mujoco XML model and initialize the system
def load_system():
    model = mujoco.MjModel.from_xml_string(xmlGLOB)
    return mjcf.load_model(model)

# Initialize the camera and perform ray tracing to render the scene
def render(sys, cam_id=1, width=128, height=128):
    cam_xmat = sys.default_camera.xmat.reshape(3, 3)
    cam_xpos = sys.default_camera.xpos
    fovy = sys.default_camera.fovy
    f = sys.default_camera.f
    h_ip = jp.tan(fovy / 2) * 2 * f
    w_ip = h_ip * (width / height)
    delta = w_ip / (2 * width)
    xs = jp.linspace(-w_ip / 2 + delta, w_ip / 2 - delta, width)
    ys = jp.flip(jp.linspace(-h_ip / 2 + delta, h_ip / 2 - delta, height))
    cam_vecs = jp.stack(jp.meshgrid(xs, ys, indexing='xy'), axis=-1)
    cam_vecs = jp.concatenate([cam_vecs, -f * jp.ones_like(cam_vecs[..., :1])], axis=-1)
    cam_vecs = jp.einsum('ij,klj->kli', cam_xmat, cam_vecs)
    cam_vecs = cam_vecs / jp.linalg.norm(cam_vecs, axis=-1, keepdims=True)

    def ray_fn(vec):
        dist, _ = mujoco.mjx.ray(sys, cam_xpos, vec, geomgroup=[True]*sys.ngeom)
        return dist

    depths = jax.vmap(ray_fn)(cam_vecs.reshape(-1, 3)).reshape(height, width)
    return depths

# Main function to setup and render the scene
def main():
    sys = load_system()
    depth = render(sys)
    Image.fromarray((depth * 100).astype('uint8'), 'L').show()

if __name__ == '__main__':
    main()
