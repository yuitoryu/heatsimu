import torch
import math
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from torch import nn
from torch.nn import functional as F
from Simulator.h2d import BC_2D, ContConduct, Heat2dSimu





if __name__ == '__main__':
    map_shape=(torch.pi, torch.pi)
    dx = 0.005
    total_time=1
    dt=0.0000125
    # dt = 0.05


    def ic(x,y):
        return torch.sin(x) * torch.sin(y)
        # return torch.sin(x)
        # return 0
    c=0.5
    plot_step=100
    
    def Q(x,y,t):
        return -torch.sin(5*x) * torch.sin(5*y) * torch.cos( torch.sqrt( (x-torch.pi/2)**2 + (y-torch.pi/2)**2) *4) * 5 * math.sin(t*torch.pi*8)
    factor = dt*c*2/dx**2
    print(factor)

    def func(x,y):
        return 0.5
    con = ContConduct(func)

    def func2(x,y,t):
        return (torch.where(y < torch.pi/4, 0, 1) + torch.where(y > 3*torch.pi/4, 0, 1) - 1)

    def func3(x,y,t):
        return (torch.where(y < torch.pi/4, 0, 1) + torch.where(y > 3*torch.pi/4, 0, 1) - 1)

        
    bc= BC_2D((1,0,func2),(1,0,func2),(1,0,func3),(1,0,func3))
    test = Heat2dSimu(map_shape, dx, total_time, dt, bc, ic, c, plot_step, Q, device='cuda', do_progress_bar=True, dtype=torch.float32)
    test.start()