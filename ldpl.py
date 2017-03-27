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

    def calculate_path_loss(self, pt, pr):
        return pt-pr

    def optimze_n(self, arg):
        pass

    def ldpl(self, pl, pl_0, n, d0, rnd):
        pl = pl_0 + 10*n*np.log10(d0/d)
        return pl

    def get_radial_distance(self, pl_d, pl_d0, n, d0):
        rand = self.create_gaussian()
        return d0*10**((pl_d-pl_d0-rand)/(10*n))

    def trilateration(self, arg):
        pass

    def create_gaussian(self):
        x = np.linspace(-4,4,100)
        return norm.pdf(x)

class residual_analysis(object):
    """docstring for residual_analysis."""
    def __init__(self, **kwargs):
        super(residual_analysis, self).__init__()

        self.error = None

        for key in ('error', 'murat'):
            if key in kwargs:
                setattr(self, key, kwargs[key])

    def calculate_errors(self, y, y_hat):
        pass

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

class visualization_tool(object):
    """docstring for visualization_tool."""
    def __init__(self):
        super(visualization_tool, self).__init__()

    def create_canvas_2d(self, arg):
        pass

    def create_canvas_1d(self, arg):
        pass

    def plot_nodes(self, arg):
        pass

    def plot_radial_distance(self, arg):
        pass

    def plot_position_estimate(self, arg):
        pass


def main(arg):
    pass

if __name__ == "__main__":
    main()
