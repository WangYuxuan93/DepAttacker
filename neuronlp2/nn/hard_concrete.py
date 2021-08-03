import numpy as np
from torch.autograd import Variable
from torch import nn
import math
import torch

class HardConcreteDist(nn.Module):

    def __init__(self, beta=2 / 3, eps=0.1, fix_temp=True):
        """
        Base class of layers using L0 Norm
        :param beta: initial temperature parameter
        :param eps: stretch widtch of s
        :param fix_temp: True if temperature is fixed
        """
        super(HardConcreteDist, self).__init__()
        #self.batch_size = batch_size
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        #self.register_buffer("uniform", torch.zeros(self.batch_size))
        self.eps = eps
        self.gamma_zeta_ratio = math.log(eps / (1.0+eps))

    def _get_prob(self, loc):
        """
        Input:
            loc: (batch_size), the location parameter
        """
        if self.training:
            batch_size = loc.size()
            u = Variable(torch.Tensor(batch_size).uniform_().to(loc.device))
            #self.uniform.uniform_()
            #u = Variable(self.uniform)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + loc) / self.temp)
            #s = s * (self.zeta - self.gamma) + self.gamma
            s = s * (1 + 2 * self.eps) - self.eps
            penalty = torch.sigmoid(loc - self.temp * self.gamma_zeta_ratio).sum()
            #print ('from training')
        else:
            #s = F.sigmoid(loc) * (self.zeta - self.gamma) + self.gamma
            s = torch.sigmoid(loc) * (1 + 2 * self.eps) - self.eps
            penalty = 0
            #print ('from test')
        return self.hard_sigmoid(s), penalty

    def hard_sigmoid(self, x):
        return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))


# Testing
"""
import matplotlib.pyplot as plt
from ipywidgets import interact
plt.style.use("ggplot")

def plot_hard_concreate(temp, eps, num=50, bins=100, **kwargs):
    hard_concrete_dist = HardConcreteDist(beta=temp, eps=0.1)
    loc = torch.Tensor(num//2).normal_()
    print (loc)

    data0 = hard_concrete_dist._get_prob(loc)
    data1 = hard_concrete_dist._get_prob(loc)

    #data = [hard_concrete(loc, temp, gamma, zeta) for _ in range(num)]
    print (data0)
    print (data1)
    data = torch.cat([data0[0],data1[0]], dim=0).numpy()
    print (data)

    bins = 10#np.arange(0,1.1,0.1)
    plt.hist(data, bins=bins, density=True, **kwargs)
    plt.show()

if __name__ == '__main__':
    temp = 0.1
    print ("Temperature:{}".format(temp))
    plot_hard_concreate(temp, eps=0.1)
"""
