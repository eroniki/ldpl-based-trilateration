#!/usr/bin/python
# coding=utf8
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


class grid_cells(object):
    """docstring for grid_cells."""

    def __init__(self, nx=25, ny=8, sx=0.9, sy=0.9, n_node=8):
        super(grid_cells, self).__init__()
        self.nx = nx
        self.ny = ny
        self.sx = sx
        self.sy = sy
        self.n_node = n_node
        self.cells = np.zeros((self.nx, self.ny, self.n_node))
        self.centers_x, self.centers_y, self.centers = \
            self.create_grid_centers(self.nx, self.ny, self.sx, self.sy)

    def mean_measurements(self, measurements, labels):
        n_obs = np.zeros_like(self.cells)
        cells = np.zeros_like(self.cells)
        for i in range(measurements.shape[0]):
            for ap in range(self.n_node):
                cells[labels[i, 0], labels[i, 1], ap] += measurements[i, ap]
                if measurements[i, ap] != 0:
                    n_obs[labels[i, 0], labels[i, 1], ap] += 1
        np.seterr(divide="ignore")
        self.cells = np.divide(cells, n_obs)
        # np.nan_to_num(self.cells)
        return self.cells

    def create_grid_centers(self, nx, ny, sx, sy):
        x = np.arange(nx)
        y = np.arange(ny)
        gcx = sx / 2 + x * sx
        gcy = sy / 2 + y * sy
        xx, yy = np.meshgrid(gcx, gcy)

        centers = np.vstack((xx.ravel(order='F'), yy.ravel(order='F'))
                            ).transpose().reshape(nx, ny, 2)
        return (gcx, gcy, centers)

    def centers():
        doc = "The centers property."

        def fget(self):
            return self._centers

        def fset(self, value):
            self._centers = value

        def fdel(self):
            del self._centers
        return locals()
    centers = property(**centers())

    def cells():
        doc = "The cells property."

        def fget(self):
            return self._cells

        def fset(self, value):
            self._cells = value

        def fdel(self):
            del self._cells
        return locals()
        cells = property(**cells())

    def nx():
        doc = "The nx property."

        def fget(self):
            return self._nx

        def fset(self, value):
            self._nx = value

        def fdel(self):
            del self._nx
        return locals()
    nx = property(**nx())

    def ny():
        doc = "The ny property."

        def fget(self):
            return self._ny

        def fset(self, value):
            self._ny = value

        def fdel(self):
            del self._ny
        return locals()
    ny = property(**ny())

    def gx():
        doc = "The gx property."

        def fget(self):
            return self._gx

        def fset(self, value):
            self._gx = value

        def fdel(self):
            del self._gx
        return locals()
    gx = property(**gx())

    def gy():
        doc = "The gy property."

        def fget(self):
            return self._gy

        def fset(self, value):
            self._gy = value

        def fdel(self):
            del self._gy
        return locals()
    gy = property(**gy())


def main():
    pass


if __name__ == "__main__":
    main()
