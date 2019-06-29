import numpy as np
import torch


def uniform(m):
    '''
    Initialization function to be applied to a neural net. Initializes
    weights to a sample from a unifrom distribution from
    [- 1 / sqrt(n), 1 / sqrt(n)] where n is the number of input features.
    '''

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        bound = 1 / np.sqrt(m.in_features)
        m.weight.data.uniform_(-bound, bound)
        m.bias.data.fill_(0)


def identity(m):
    '''
    Initialization function for linear layers, setting weights to the identity
    matrix and bias to zero.
    '''

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.eye_(tensor=m.weight)
        m.bias.data.fill_(0)


def rotation(m):
    '''
    Initialization function for linear layers, setting weights to a small
    rotation and bias to zero.
    '''

    theta = np.pi / 4
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data = torch.Tensor([[np.cos(theta), -np.sin(theta)],
                                      [np.sin(theta), np.cos(theta)]])
        m.bias.data.fill_(0)
