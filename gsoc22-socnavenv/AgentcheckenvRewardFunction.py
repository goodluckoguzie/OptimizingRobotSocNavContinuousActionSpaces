from calendar import c
import time
import gym
import numpy as np

import time
import gym
import numpy as np
import socnavenv
import os
import pygame
import numpy as np 
import matplotlib.pyplot as plt
import sys
import argparse

env = gym.make("SocNavEnv-v1")
env.configure("../configs/env_timestep_1.yaml")
env.reset()
env.render()




from simplejson import load
import os
import pygame

import numpy as np 
import matplotlib.pyplot as plt


pygame.init()
pygame.joystick.init()
controller = pygame.joystick.Joystick(0)


axis_data = { 0:0, 1:0}
button_data = {}
hat_data = {}


episodes = 50



def axis_data_to_action(axis_data):
    return np.array([-axis_data[1], -axis_data[0]])    
AverageReward = 0

for episode in range(episodes):
    done = False
    obs = env.reset()

    rewards = []
    cumulative_reward =  []
    R = 0
    print("Current episode  = ",episode)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                axis_data[event.axis] = round(event.value,2)
            elif event.type == pygame.JOYBUTTONDOWN:
                button_data[event.button] = True
            elif event.type == pygame.JOYBUTTONUP:
                button_data[event.button] = False
            elif event.type == pygame.JOYHATMOTION:
                hat_data[event.hat] = event.value

        # Insert your code on what you would like to happen for each event here!
        action = axis_data_to_action(axis_data)

        # print("action", action)
        obs, reward, done, info = env.step(action)
        # print('reward',reward)
        R = R + reward
        env.render()


        rewards.append(reward)
        cumulative_reward.append(R)
        
        
    print('Total reward',R)
    AverageReward = AverageReward + R
    np.save('AverageReward.npy', AverageReward) 
    # r = np.array(rewards)
    # plt.axhline(y=0., color='k', linestyle='-')
    # plt.plot(r) 
    # plt.plot(np.array(cumulative_reward))
    #     # plt.ylim([-0.02, 0.02])
    #     # plt.yticks([-0.02, -0.01, 0, 0.01, 0.02])
    #     # #plt.pause(socnavenv1.TIMESTEP)
    #     # plt.pause(socnavenv.TIMESTEP)


    # plt.show()


print("Total reward = ",AverageReward)
print("Average reward after 50 episodes = ",AverageReward/episodes)

env.close()