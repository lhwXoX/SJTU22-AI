#CNN模型
import jittor as jt
from jittor import nn, Module
import numpy as np
import sys, os
import random
import math
from jittor import init

class Model (Module):
    def __init__ (self):
        super (Model, self).__init__()
        self.mod = nn.Sequential(
            nn.Conv(3, 32, 3, 1, 1),
            nn.BatchNorm(32),
            nn.Relu(),           
            nn.Conv(32, 32, 3, 1, 1),        
            nn.BatchNorm(32),
            nn.Relu(),           
            nn.MaxPool2d(2,2),
            nn.Conv(32, 64, 3, 1, 1),           
            nn.BatchNorm(64),
            nn.Relu(),           
            nn.Conv(64, 64, 3, 1, 1),    
            nn.BatchNorm(64),
            nn.Relu(),
            nn.MaxPool2d(2,2),
            nn.Conv(64, 128, 3, 1, 1),    
            nn.BatchNorm(128),
            nn.Relu(),           
            nn.Conv(128, 128, 3, 1, 1), 
            nn.BatchNorm(128),
            nn.Relu(),
            nn.AvgPool2d(2,2),
            nn.Flatten(),
            nn.Linear(128*4*4, 512),   
            nn.BatchNorm(512),
            nn.Relu(),
            nn.Linear(512, 10)
        )

    def execute (self, x) :
        x = self.mod(x)
        return x