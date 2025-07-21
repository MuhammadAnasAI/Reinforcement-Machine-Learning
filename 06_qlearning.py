import gym
import numpy as np
import matplotlib.pyplot as plt

# Create the environment
env = gym.make("MountainCar-v0", render_mode="human")  # or "rgb_array" if you prefer

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
SHOW_EVERY = 500

# Discretize the observation space
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Exploration settings
epsilon = 1.0  # Starting epsilon for exploration
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Initialize Q-table
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Track episode rewards
episode_rewards = []
aggre_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    """Convert continuous state to discrete state."""
    state = np.array(state)  # Ensure state is a NumPy array
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))

for episode in range(EPISODES):
    episodes_reward = 0
    if episode % SHOW_EVERY == 0:
        print(f"Episode {episode}")
        render = True
    else:
        render = False

    # Reset the environment
    initial_state = env.reset()  # Get the initial state
    discrete_state = get_discrete_state(initial_state[0])  # Extract the first element, which is the state
    done = False

    while not done:
        # Choose action based on exploration-exploitation tradeoff
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        # Step the environment
        new_state, reward, done, trancated, info = env.step(action)  # Unpack four values

        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        # Update Q-table
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0  # If goal is reached, set Q-value to 0

        discrete_state = new_discrete_state
        episodes_reward += reward  # Accumulate the reward

    episode_rewards.append(episodes_reward)  # Store total reward for this episode

    # Decay epsilon
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # Calculate and store the average, min, and max rewards for plotting
    if episode % SHOW_EVERY == 0 and episode > 0:
        average_reward = sum(episode_rewards[-SHOW_EVERY:]) / len(episode_rewards[-SHOW_EVERY:])
        aggre_ep_rewards['ep'].append(episode)
        aggre_ep_rewards['avg'].append(average_reward)
        aggre_ep_rewards['min'].append(min(episode_rewards[-SHOW_EVERY:]))
        aggre_ep_rewards['max'].append(max(episode_rewards[-SHOW_EVERY:]))

        print(f"Episode: {episode} avg: {average_reward:.2f} min: {min(episode_rewards[-SHOW_EVERY:])} max: {max(episode_rewards[-SHOW_EVERY:])}")

env.close()

# Plotting the results
plt.plot(aggre_ep_rewards['ep'], aggre_ep_rewards['avg'], label="Average Reward")
plt.plot(aggre_ep_rewards['ep'], aggre_ep_rewards['min'], label="Min Reward")
plt.plot(aggre_ep_rewards['ep'], aggre_ep_rewards['max'], label="Max Reward")
plt.legend(loc='upper left')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Episode Rewards Over Time')
plt.show()