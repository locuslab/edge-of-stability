import torch
from torch.utils.data import TensorDataset
from math import sqrt
from numpy.polynomial import chebyshev
import numpy as np


def make_chebyshev_dataset(k, n=10000):
    """
    Generate a dataset of n points evenly spaced on the interval [-1, 1], labeled by the chebyshev polynomial of
    degree k.
    """
    X = torch.linspace(-1, 1, n)
    c = np.zeros(k + 1)
    c[-1] = 1
    y = torch.from_numpy(chebyshev.chebval(X.numpy(), c)).float()
    dataset = TensorDataset(X.unsqueeze(1), y.unsqueeze(1))
    return dataset, dataset


def make_linear_dataset(n, d, seed=0):
    """
    Create a dataset for training a deep linear network with n datapoints of dimension d.
    """
    torch.manual_seed(seed)
    X = (torch.qr(torch.randn(n, d))[0] * sqrt(n)).cuda()
    A = torch.randn(d, d).cuda()
    Y = X.mm(A.t())
    return TensorDataset(X, Y), TensorDataset(X, Y)


