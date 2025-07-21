import gymnasium as gym
import time
import numpy as np

# Initialize the environment with the correct render mode
env = gym.make("MountainCar-v0", render_mode="human")

# Initialize the learning rate, discount, episodes:
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000

# Create the Discrete OS size with observation values:
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

# Create the Discrete OS window size (step size per bucket):
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
#Add the eplison value = 0.5:
epsilon = 5 
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Create a Q-table with random values (size based on state space and action space):
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Create a function to convert the continuous state to a discrete state:
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))
for episodes in range (EPISODES):
    if episodes % SHOW_EVERY == 0:
        print(f"Episode {episodes}")
        render = True
    else:
        render = False
# Reset the environment and get the initial state:
    initial_state = env.reset()[0]  # Extract the first element from the tuple returned by reset
    discrete_state = get_discrete_state(initial_state)

#print(discrete_state)
#Print the q_table with discrete_state:
#print(np.argmax(q_table[discrete_state]))
# Run the environment
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)     # Always accelerate to the right
        new_state, reward, done, truncated, info = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state [0] >= env.goal_position:
            q_table [discrete_state + (action,)] = 0
        discrete_state = new_discrete_state
    #print(reward, new_state)
    if END_EPSILON_DECAYING >= episodes >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
    # Optional delay for smoother rendering
        time.sleep(0.02)

# Close the environment
env.close() 
