import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory for saving Q-tables if it doesn't exist
if not os.path.exists("qtables"):
    os.makedirs("qtables")

# Initialize the environment with the correct render mode
env = gym.make("MountainCar-v0", render_mode="human")

# Initialize the learning rate, discount, episodes:
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
SHOW_EVERY = 500

# Create the Discrete OS size with observation values:
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

# Create the Discrete OS window size (step size per bucket):
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Epsilon parameters
epsilon = 5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Create a Q-table with random values
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Add the number of episode rewards
episode_rewards = []

# Add the number of episode steps:
aggre_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

# Create a function to convert the continuous state to a discrete state:
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))

for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        print(f"Episode {episode}")
        render = True
    else:
        render = False

    # Reset the environment and get the initial state:
    initial_state = env.reset()[0]
    discrete_state = get_discrete_state(initial_state)

    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)  # Random action

        # Unpack the 5 values returned by step
        new_state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        if not done:
            # Update Q-table with Q-learning formula
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            # If goal is achieved, set Q-value to 0
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Decay epsilon value
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # Store rewards
    episode_rewards.append(episode_reward)

    # Save Q-tables every 10 episodes
    if episode % 10 == 0:
        np.save(f"qtables/{episode}-qtable.npy", q_table)

    # Aggregate statistics every SHOW_EVERY episodes
    if episode % SHOW_EVERY == 0:
        average_reward = sum(episode_rewards[-SHOW_EVERY:]) / len(episode_rewards[-SHOW_EVERY:])
        aggre_ep_rewards['ep'].append(episode)
        aggre_ep_rewards['avg'].append(average_reward)
        aggre_ep_rewards['min'].append(min(episode_rewards[-SHOW_EVERY:]))
        aggre_ep_rewards['max'].append(max(episode_rewards[-SHOW_EVERY:]))

        print(f"Episode: {episode} avg: {average_reward:.2f} min: {min(episode_rewards[-SHOW_EVERY:]):.2f} max: {max(episode_rewards[-SHOW_EVERY:]):.2f}")

# Close the environment
env.close()

# Plot the aggregated rewards
plt.plot(aggre_ep_rewards['ep'], aggre_ep_rewards['avg'], label="avg")
plt.plot(aggre_ep_rewards['ep'], aggre_ep_rewards['min'], label="min")
plt.plot(aggre_ep_rewards['ep'], aggre_ep_rewards['max'], label="max")
plt.legend(loc="best")
plt.title("MountainCar - Q Learning Performance")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.show()
