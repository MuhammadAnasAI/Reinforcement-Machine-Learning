import gymnasium as gym
import time
import numpy as np

# Initialize the environment with the correct render mode
env = gym.make("MountainCar-v0", render_mode="human")

# Reset the environment
env.reset()
#Print the Observation space with high and low values into env:
print(env.observation_space.high)
print(env.observation_space.low)
#Print the Action space with high and low values into env:
print(env.action_space.n)
#Create the Discrete OS size with observations values:
DISCRETE_OS_SIZE = [20] * len (env.observation_space.high)
#Create the Discrete OS win values with observations values:
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
#Print the values:
print(discrete_os_win_size)
#Create a q_table with possible combinations and outcomes with reward values (Size is also created):
q_table = np.random.uniform(low= -2, high=0, size= (DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table.shape)
print(q_table)
'''
# Run the environment
done = False
while not done:
    action = 2  # Always accelerate to the right
    new_state, reward, done, truncated, info = env.step(action)
    print(reward, new_state)
    
    # Optional delay for smoother rendering
    time.sleep(0.02)

# Close the environment
env.close() '''

