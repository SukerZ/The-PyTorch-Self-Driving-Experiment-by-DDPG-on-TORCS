import pdb
import torch
from torch.autograd import Variable
import torch.nn as nn
import os

HIDDEN1_UNITS = 300; HIDDEN2_UNITS = 600

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_size, HIDDEN1_UNITS),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(action_dim, HIDDEN2_UNITS)
        self.fc3 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.fc4 = nn.Sequential(
            nn.Linear(HIDDEN2_UNITS, HIDDEN2_UNITS),
            nn.ReLU()
        )
        self.fc5 = nn.Linear(HIDDEN2_UNITS, 1)
        
    def forward(self, s,a):
        s = self.fc1(s); s = self.fc3(s)
        a = self.fc2(a); x = a+s
        x = self.fc4(x);
        return self.fc5(x)

class CriticBrain(object):
    def save(self):
        print("Save critic net parameters.")
        torch.save(self.net.state_dict(),"critic.pth")
        
    def load(self):
        if os.path.exists("critic.pth"):
            print("Load critic net parameters.")
            self.net.load_state_dict(torch.load("critic.pth"))
            self.tnet.load_state_dict(torch.load("critic.pth"))
            
    def __init__(self, state_size, action_dim):
        self.net, self.tnet = CriticNetwork(state_size, action_dim), CriticNetwork(state_size, action_dim)
        self.load()
        self.func = nn.MSELoss()
        lrc = 1e-3
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lrc)
    
    def dafen(self,state,action):
        s = torch.Tensor(state)
        a = torch.Tensor(action)
        return self.tnet(s,a)
    
    def predict(self,state,action):
        tmp = []
        for i in range(len(action)):
            e = [action[i][0][0], action[i][1][0], action[i][2][0]]
            tmp.append(e)
        s = torch.Tensor(state)
        a = torch.Tensor(tmp)
        return self.net(s,a)
        
    def train(self, yuce, zhenshi, tau):
        zhenshi = torch.Tensor(zhenshi)
        self.optimizer.zero_grad()
        loss = self.func(yuce, zhenshi)
        loss.backward()
        self.optimizer.step()
        weights = self.net.state_dict()
        tweights = self.tnet.state_dict()
        for k in tweights.keys():
            tweights[k] = weights[k] * tau + (1 - tau) * weights[k]
        
        return loss
        
        
        
        
        
        
