#!/usr/bin/python
from __future__ import division

import numpy as np
import numpy.matlib
import scipy as sp
from scipy.stats import norm
from scipy.optimize import minimize_scalar, minimize, least_squares
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ldpl_based_trilateration(object):
    """docstring for ldpl_based_trilateration."""
    def __init__(self, measurement, grid_labels, pos_node, pt=24, pr_d0=0, d0=0, d=0):
        super(ldpl_based_trilateration, self).__init__()
        self.measurement = measurement
        self.grid_labels = grid_labels
        self.pt = pt
        self.pl_d0 = self.pt - pr_d0;
        self.pos_node = pos_node
        self.mean_measurements = None
        self.pl = self.calculate_path_loss(self.pt, self.measurement)
        self.d0 = d0
        self.d = d;
        self.n_data, self.n_node = self.measurement.shape
        self.n_samples = 100;
        self.rand = self.create_gaussian()

    def calculate_path_loss(self, pt, pr):
        return pt-pr

    def optimize_n(self, n_test):
        loss = np.zeros((n_test.size))
        # mean_loss = np.zeros((n_test.size), dtype=np.float64)
        print "PL", self.pl.shape
        print "PL_d0", self.pl_d0.shape
        # for ni in range(5):
        for ni in range(n_test.size):
            l = 0
            for i in range(self.n_data):
                d_hat = self.get_radial_distance(self.pl[i,:], self.pl_d0, n_test[ni], self.d0)
                l_i_j = self.loss(self.d[i,:], d_hat)
                l_i= np.nanmin(l_i_j, axis=0)
                l += np.sum(l_i)
                # print n_test.size, d_hat.shape, l_i_j.shape, l_i.shape, l_i
            print " ".join(("n:", str(n_test[ni]), "%:", str((ni+1)/n_test.size), "loss:", str(l)))
            loss[ni]=l/self.n_data
        return loss

    def optimize_n_least_squares(self):
        return least_squares(self.loss_least_squares, x0=0.5)

    def loss(self, d, d_hat):
        return (d-d_hat)**2

    def ldpl(self, pl, pl_0, n, d0, rnd):
        pl = pl_0 + 10*n*np.log10(d0/d)
        return pl

    def get_radial_distance(self, pl_d, pl_d0, n, d0):
        # print rand.shape
        rand = np.matlib.repmat(self.rand, pl_d0.size, 1).transpose()
        # print rand.shape
        # print "pl", pl_d.shape
        # print "pl_d0", pl_d0.shape
        # print "n", n
        # print "d0", d0, "d0 shape", d0.shape
        # return d0*10**((pl_d-pl_d0-rand)/(10*n))
        # print .shape
        return (d0*10**((pl_d-pl_d0-rand)/(10*n)))

    def trilateration(self, arg):
        pass

    def create_gaussian(self):
        x = np.linspace(-4,4,self.n_samples)
        return norm.pdf(x)

    ''' Properties '''
    def d():
        doc = "The d property."
        def fget(self):
            return self._d
        def fset(self, value):
            self._d = value
        def fdel(self):
            del self._d
        return locals()
    d = property(**d())

    def d0():
        doc = "The d0 property."
        def fget(self):
            return self._d0
        def fset(self, value):
            self._d0 = value
        def fdel(self):
            del self._d0
        return locals()
    d0 = property(**d0())

    def pl_d0():
        doc = "The pl_d0 property."
        def fget(self):
            return self._pl_d0
        def fset(self, value):
            self._pl_d0 = value
        def fdel(self):
            del self._pl_d0
        return locals()
    pl_d0 = property(**pl_d0())

    def pl():
        doc = "The pl property."
        def fget(self):
            return self._pl
        def fset(self, value):
            self._pl = value
        def fdel(self):
            del self._pl
        return locals()
    pl = property(**pl())

    def measurement():
        doc = "The measurement property."
        def fget(self):
            return self._measurement
        def fset(self, value):
            self._measurement = value
        def fdel(self):
            del self._measurement
        return locals()
    measurement = property(**measurement())

    def grid_labels():
        doc = "The grid_labels property."
        def fget(self):
            return self._grid_labels
        def fset(self, value):
            self._grid_labels = value
        def fdel(self):
            del self._grid_labels
        return locals()

    grid_labels = property(**grid_labels())

    def pt():
        doc = "The pt property."
        def fget(self):
            return self._pt
        def fset(self, value):
            self._pt = value
        def fdel(self):
            del self._pt
        return locals()
    pt = property(**pt())

class grid_cells(object):
    """docstring for grid_cells."""
    def __init__(self, nx=25, ny=8, sx=0.9, sy=0.9, n_node = 8):
        super(grid_cells, self).__init__()
        self.nx = nx
        self.ny = ny
        self.sx = sx
        self.sy = sy
        self.n_node = n_node
        self.cells = np.zeros((self.nx, self.ny, self.n_node))
        self.centers_x, self.centers_y, self.centers = self.create_grid_centers(self.nx, self.ny, self.sx, self.sy)

    def mean_measurements(self, measurements, labels):
        n_obs = np.zeros_like(self.cells)
        cells = np.zeros_like(self.cells)
        for i in range(measurements.shape[0]):
            for ap in range(self.n_node):
                cells[labels[i,0], labels[i,1], ap] += measurements[i, ap]
                if measurements[i, ap] != 0:
                    n_obs[labels[i,0], labels[i,1], ap] += 1
        np.seterr(divide="ignore")
        self.cells = np.divide(cells, n_obs)
        # np.nan_to_num(self.cells)
        return self.cells

    def create_grid_centers(self, nx, ny, sx, sy):
        x = np.arange(nx)
        y = np.arange(ny)
        gcx = sx/2 + x*sx
        gcy = sy/2 + y*sy
        xx, yy = np.meshgrid(gcx, gcy)

        centers = np.vstack((xx.ravel(order='F'), yy.ravel(order='F'))).transpose().reshape(nx,ny,2)
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
