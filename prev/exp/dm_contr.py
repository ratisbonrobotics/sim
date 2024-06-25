import numpy as np
import imageio
from dm_contr import suite

# Load the environment
env = suite.load(domain_name="cartpole", task_name="swingup")

# Define the action specification
action_spec = env.action_spec()

# Prepare the video writer
with imageio.get_writer('simulation.mp4', fps=60) as video:
    for _ in range(200):  # Adjust the number of frames or use a more dynamic condition
        action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        time_step = env.step(action)
        frame = env.physics.render(height=480, width=640, camera_id=0)  # Render a frame
        video.append_data(frame)  # Append the frame to the video

        if time_step.last():
            break

print("Video saved successfully.")