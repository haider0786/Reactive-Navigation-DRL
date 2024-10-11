import numpy as np
import pandas as pd
data = np.load('/home/haider/DRL-robot-navigation/TD3/GAIL/results_1/episode_reward.npy')

sardns_data = pd.DataFrame(data)
filename = "average_reward_episode"
sardns_data.to_csv( "/home/haider/DRL-robot-navigation/TD3/GAIL/results_1/average_reward_episode.csv")