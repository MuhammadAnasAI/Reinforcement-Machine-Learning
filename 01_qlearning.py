import gymnasium as gym
import matplotlib.pyplot as plt
env = gym.make('CartPole', render_mode='rgb_array')
state, info = env.reset(seed=42)
print(state)  
#Create a graph of the state_image:
def render():

    state_image = env.render()
    plt.imshow(state_image)
    plt.show()
render()
#Performing actions : moving left = 0, moving right = 1
action = 1
#Performing the action and getting the next state and reward
state, reward, terminated, truncated, info = env.step(action)
print("State :", state)
print("Rewards:", reward)
print("Terminated:", terminated)
while not terminated:
    action = 1
    state, reward, terminated, _, _ = env.step(action)
render()    

