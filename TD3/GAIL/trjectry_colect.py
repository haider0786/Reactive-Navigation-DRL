import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from velodyne_test import GazeboEnv

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        # print("s_shaope", s.shape)
        s = F.relu(self.layer_2(s))
        # print("s_shaope", s.shape)
        a = self.tanh(self.layer_3(s))
        # print("a_shaope", a.shape)
        return a


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        # Function to load network parameters
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
# print("device", device)
seed = 0  # Random seed number
max_ep = 500  # maximum number of steps per episode
file_name = "TD3_velodyne"  # name of the file to load the policy from


# Create the testing environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2

# Create the network
network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "./models_1")
except:
    raise ValueError("Could not load the stored model parameters")

done = False
episode_timesteps = 0
state = env.reset()
goal1 = -1.2
goal2 = 0.171
goal_num = 2
# dg = env.onestep(goal1, goal2)
dg = 2.0
done_step =0
results_app = []
j = 0
# Begin the testing loop
start = time.time()
sardns_data = 0
while(1):
    action = network.get_action(np.array(state))

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target, dis_r1r2, dist, length, gx, gy, rx, ry, joy_en = env.step(a_in)
    done_step = done_step + int(target)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    # results_save = list([rx, ry]), list([gx, gy]) + list([dis_r1r2]) + list([int(done)]) \
    #                + list([length])
    if joy_en == 1:
        results_save = list([rx]) + list([ry])
        results_app.append(results_save)
        sardns_data = pd.DataFrame(results_app)
        sardns_data.to_csv('/home/haider/DRL-robot-navigation/TD3/GAIL/trained_models/static_dynamic/headon_new/escrl2_2h' + ".csv")
    try:
        if dg <= 0.05 and goal_num == 2:
            goal1 = 1.2
            goal2 = 0.171
        #     goal_num = 1
        #     print("goal_num=2", goal_num)
        # if dg <= 1 and goal_num == 1:
        #     goal1 = -1.2
        #     goal2 = 0.171
        #     goal_num = 2
        # if goal_num == 1:
        #     break
        # print("goal_num dg", goal_num, dg)
        # print("goal1, goal2", goal1, goal2)
        # if j > 50:
        #     # dg = env.onestep(goal1, goal2)
        # j = j+1

        # On termination of episode
        if done_step > 10:
            print("done_step", done_step)
            # break
        if done:
            state = env.reset()
            done = False
            episode_timesteps = 0
            goal1 = -1.2
            goal2 = 0.171
            goal_num = 2
        else:
            state = next_state
    except KeyboardInterrupt:
        break

