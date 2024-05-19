import numpy as np
import imageio
from dm_control import suite

# Load the environment
env = suite.load(domain_name="cartpole", task_name="swingup", visualize_reward=True)

# Define the action specification
action_spec = env.action_spec()

# Define the hyperparameters
num_iterations = 10
num_episodes_per_iteration = 50
num_best_episodes = 10
num_steps_per_episode = 200

# Initialize the mean and standard deviation for action distribution
mean = np.zeros(action_spec.shape)
std_dev = np.ones(action_spec.shape)

# Prepare the video writer
with imageio.get_writer('simulation.mp4', fps=60) as video:
    for iteration in range(num_iterations):
        episode_rewards = []
        episode_actions = []

        for episode in range(num_episodes_per_iteration):
            time_step = env.reset()
            episode_reward = 0
            episode_action = np.zeros((num_steps_per_episode, *action_spec.shape))

            for step in range(num_steps_per_episode):
                action = np.random.normal(mean, std_dev)
                action = np.clip(action, action_spec.minimum, action_spec.maximum)
                time_step = env.step(action)
                episode_reward += time_step.reward
                episode_action[step] = action

                if time_step.last():
                    break

            episode_rewards.append(episode_reward)
            episode_actions.append(episode_action)

        # Sort episodes by reward
        best_episode_indices = np.argsort(episode_rewards)[-num_best_episodes:]

        # Update mean and standard deviation based on best episodes
        best_actions = [episode_actions[i] for i in best_episode_indices]
        mean = np.mean(best_actions, axis=(0, 1))
        std_dev = np.std(best_actions, axis=(0, 1))

        print(f"Iteration {iteration + 1}: Mean Reward = {np.mean(episode_rewards):.2f}")

    # Run the best policy and record the video
    time_step = env.reset()
    while not time_step.last():
        action = mean
        time_step = env.step(action)
        frame = env.physics.render(height=480, width=640, camera_id=0)
        video.append_data(frame)

print("Video saved successfully.")