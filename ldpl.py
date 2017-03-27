#!/usr/bin/python
from __future__ import division

import numpy as np
import numpy.matlib
import scipy as sp
import os

from sklearn import mixture
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ldpl_based_trilateration(object):
    """docstring for ldpl_based_trilateration."""
    def __init__(self, measurement, grid_labels, pos_node, pt=24, **kwargs):
        super(ldpl_based_trilateration, self).__init__()
        self.measurement = measurement
        self.grid_labels = grid_labels
        self.pt = pt
        self.pos_node = pos_node
        self.mean_measurements = None

    def optimze_n(self, arg):
        pass

    def ldpl(self, pr, pr_0, n, d0, rnd):
        pr = pr_0 + 10*n*np.log10(d0/d)
        return pr

    def get_radial_distance(self, arg):
        pass

    def trilateration(self, arg):
        pass

    def create_gaussian(self):
        x = np.linspace(-4,4,100)
        return norm.pdf(x)

class residual_analysis(object):
    """docstring for residual_analysis."""
    def __init__(self, y, y_hat):
        super(residual_analysis, self).__init__()
        self.y = y
        self.y_hat = y_hat

    def localization_cdf(self, arg):
        pass

    def localization_error(self, y, y_hat):
        pass

    def mahalonobis_distance(self, arg):
        # scipy.spatial.distance.mahalanobis(u, v, VI)
        pass

    def euclidean_distance(self, arg):
        # scipy.spatial.distance.euclidean(u, v)
        pass


def main(arg):
    pass

if __name__ == "__main__":
    main()
