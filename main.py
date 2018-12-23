import pdb
from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from ActorNetwork import ActorBrain
from CriticNetwork import CriticNetwork
from CriticNetwork import CriticBrain
from OU import OU
import timeit

import torch
from torch.autograd import Variable
import torch.nn as nn

OU = OU()
max_steps = 100000
BUFFER_SIZE = 100000 
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.001
LRA = 0.0001
LRC = 0.001
action_dim = 3
state_dim = 29
vision = False
EXPLORE = 100000.
episode_count = 2000
max_steps = 100000
reward = 0
done = False
step = 0
epsilon = 1
tau = 0.001
update = 100

if __name__ == "__main__":
    np.random.seed(1337)
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    env.reset(relaunch=True)
    actor = ActorBrain(state_dim)
    critic = CriticBrain(state_dim, action_dim)
    buff = ReplayBuffer(BUFFER_SIZE)
    
    for i in range(episode_count):
        print("Episode: " + str(i))
        ob = env.reset(relaunch = True)
            
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)).tolist()
        total_reward = 0.
        
        for j in range(max_steps):
            loss = 0
            a_t = actor.getaction(s_t)
            fenshu = critic.dafen(s_t,a_t)
            noise0 = max(epsilon, 0) * OU.function(a_t[0],  0.0 , 0.60, 0.30)
            noise1 = max(epsilon, 0) * OU.function(a_t[1],  0.5 , 1.00, 0.10)
            noise2 = max(epsilon, 0) * OU.function(a_t[2], -0.1 , 1.00, 0.05)
            
            a_t[0] = a_t[0] + noise0
            a_t[1] = a_t[1] + noise1
            a_t[2] = a_t[2] + noise2
            
            ob, r_t, done, info = env.step(a_t)
            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)).tolist()
            buff.add(s_t, a_t, r_t, s_t1, done)
            
            batch = buff.getBatch(BATCH_SIZE)
            states = [e[0] for e in batch]
            actions = [e[1] for e in batch]
            rewards = [e[2] for e in batch]
            new_states = [e[3] for e in batch]
            dones = [e[4] for e in batch]
            y_t = np.zeros((len(batch)))
            
            new_a = actor.BatchGetaction(new_states)
            target_q_values = critic.dafen(new_states, new_a)
            
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k].item()
           
            yuce = critic.predict(states, actions)
            loss += critic.train(yuce, y_t, tau)
            actor.train(fenshu, tau)
            
            total_reward += r_t
            s_t = s_t1
            print("Episode: ",i,"Step: ",step,"Steering: ",a_t[0][0], "Acceleration: ",a_t[1][0],"Break",a_t[2][0],"Reward: ",r_t, "Loss: ",loss.item())
            step += 1
            if done:
                break;
        
        if i%update==0 and i>0:
            actor.save(); actor.load()
            critic.save(); critic.load()
        
        print("Total Reward @" + str(i) + "-th Episode :Reward " + str(total_reward))
        print("Total Step: " + str(step) + '\n')
    env.reset()
    env.end()
    print("Finish.")
