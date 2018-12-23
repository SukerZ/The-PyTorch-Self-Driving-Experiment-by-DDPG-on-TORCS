import pdb
import torch
from torch.autograd import Variable
import torch.nn as nn
import os

class ActorNetwork(nn.Module):
    def __init__(self,state_size):
        super(ActorNetwork,self).__init__()
        self.HIDDEN1_UNITS = 300
        self.HIDDEN2_UNITS = 600
        self.fc1 = nn.Sequential(
            nn.Linear(state_size, self.HIDDEN1_UNITS),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.HIDDEN1_UNITS, self.HIDDEN2_UNITS),
            nn.ReLU()
        )
        self.out1 = nn.Sequential(
            nn.Linear(self.HIDDEN2_UNITS, 1),
            nn.Tanh()
        )
        self.out2 = nn.Sequential(
            nn.Linear(self.HIDDEN2_UNITS, 1),
            nn.Sigmoid()
        )
        self.out3 = nn.Sequential(
            nn.Linear(self.HIDDEN2_UNITS, 1),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        x = self.fc1(x); x = self.fc2(x)
        steering = self.out1(x)
        acceleration = self.out2(x)
        brake = self.out3(x)
        return [steering.item(),acceleration.item(),brake.item()]      

class ActorBrain(object):
    def save(self):
        print("Save actor net parameters.")
        torch.save(self.net.state_dict(),"actor.pth")
        
    def load(self):
        if os.path.exists("actor.pth"):
            print("Load actor net parameters.")
            self.net.load_state_dict(torch.load("actor.pth"))
            self.tnet.load_state_dict(torch.load("actor.pth"))

    def __init__(self,state_size):
        self.net, self.tnet = ActorNetwork(state_size), ActorNetwork(state_size)
        self.load()
        lra = 1e-4
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lra)
        
    def train(self,fenshu,tau):
        self.optimizer.zero_grad()
        loss = -fenshu
        loss.backward()
        self.optimizer.step()
        weights = self.net.state_dict()
        tweights = self.tnet.state_dict()
        for k in tweights.keys():
            tweights[k] = weights[k] * tau + (1 - tau) * weights[k]
        
    def getaction(self, currentstate):
        tmp = torch.Tensor(currentstate)
        return self.net(tmp)
        
    def BatchGetaction(self,states):
        tmp = []
        for e in states:
            tmp.append(self.tnet(torch.Tensor(e)))
        return tmp
    
        
        
        
            
        
   
        
