#!/usr/bin/python
from __future__ import division

import numpy as np
import numpy.matlib
import scipy as sp
from scipy.stats import norm
from scipy.optimize import minimize_scalar, minimize, least_squares, curve_fit
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

class visualization_tool(object):
    """docstring for visualization_tool."""
    def __init__(self):
        super(visualization_tool, self).__init__()
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        plt.minorticks_on()
        # plt.show()

    def plot_loss(self, n, loss):
        #
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        _ = self.ax.plot(n, loss)
        # return fig

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


def main():
    pass

if __name__ == "__main__":
    main()
