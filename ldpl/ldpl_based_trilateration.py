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


class ldpl_based_trilateration(object):
    """docstring for ldpl_based_trilateration."""

    def __init__(self,
                 measurement,
                 grid_labels,
                 pos_node,
                 pt=24,
                 pr_d0=0,
                 d0=0,
                 d=0):
        super(ldpl_based_trilateration, self).__init__()
        self.measurement = measurement
        self.grid_labels = grid_labels
        self.pt = pt
        self.pl_d0 = self.pt - pr_d0
        self.pos_node = pos_node
        self.mean_measurements = None
        self.pl = self.calculate_path_loss(self.pt, self.measurement)
        self.d0 = d0
        self.d = d
        self.n_data, self.n_node = self.measurement.shape
        self.n_samples = 1000
        self.rand = self.create_gaussian()

    def calculate_path_loss(self, pt, pr):
        return pt - pr

    def optimize_n(self, n_test):
        loss = np.zeros((n_test.size))
        # mean_loss = np.zeros((n_test.size), dtype=np.float64)
        print "PL", self.pl.shape
        print "PL_d0", self.pl_d0.shape
        # for ni in range(5):
        for ni in range(n_test.size):
            l = 0
            for i in range(self.n_data):
                d_hat = self.get_radial_distance(
                    self.pl[i, :], self.pl_d0, n_test[ni], self.d0)
                l_i_j = self.loss(self.d[i, :], d_hat)
                l_i = np.nanmin(l_i_j, axis=0)
                l += np.sum(l_i)
                # print n_test.size, d_hat.shape, l_i_j.shape, l_i.shape, l_i
            print " ".join(("n:", str(n_test[ni]), "%:",
                            str((ni + 1) / n_test.size), "loss:", str(l)))
            loss[ni] = l / self.n_data
        return loss

    def curve_fit(self):
        print " ".join(("pl_d: ", str(self.pl.shape), "pl_d0:",
                        str(self.pl_d0.shape), "d0: ", str(self.d0.shape)))
        pl_d0_rep = np.matlib.repmat(self.pl_d0, 1543, 1)
        d0_rep = np.matlib.repmat(self.d0, 1543, 1)
        print " ".join(("pl_d: ", str(self.pl.shape), "pl_d0_rep:",
                        str(pl_d0_rep.shape), "d0_rep: ", str(d0_rep.shape)))
        pl_d0_rep = pl_d0_rep.ravel()
        d0_rep = d0_rep.ravel()
        pl_rep = self.pl.ravel()
        print " ".join(("pld: ", str(pl_rep.shape), "pld0:",
                        str(pl_d0_rep.shape), "d0_rep:", str(d0_rep.shape)))
        xdata = np.array([pl_rep, pl_d0_rep, d0_rep])
        print " ".join(("xdata:", str(xdata.shape)))
        # self.d_hat(xdata, 6)
        popt, pcov = curve_fit(self.d_hat, xdata, self.d.ravel())
        return popt, pcov

    def d_hat(self, x, n):
        '''  x = [pl_d (nx8), pl_d0 (nx8), d0 (nx8)] '''
        pl_d = x[0, :].ravel()
        pl_d0 = x[1, :].ravel()
        d0 = x[2, :].ravel()
        rand = np.matlib.repmat(self.rand, 12344, 1).transpose()
        print " ".join(("d_hat", "pl_d:", str(pl_d.shape), "pl_d0:",
                        str(pl_d0.shape), "d0:", str(d0.shape), "rand",
                        str(rand.shape)))
        d_hat = np.nanmin(d0 * 10**((pl_d - pl_d0 - rand) / (10 * n)), axis=0)
        print np.count_nonzero(np.isnan(d_hat)), d_hat.shape
        return d_hat

    ''' TODO: Gotta think about it thoroughly! '''

    def jac(self, x, n):
        ''' x = [pl_d, pl_d0, d0] '''
        rand = self.rand.reshape((self.n_samples, 1))
        dpld = x[2, :] * np.log(10) / (10 * n) * \
            10**((x[0, :] - x[1, :] - rand) / (10 * n))
        dpld0 = -1 * x[2, :] * np.log(10) / (10 * n) * \
            10**((x[0, :] - x[1, :] - rand) / (10 * n))
        dd0 = 10**((x[0, :] - x[1, :] - rand) / (10 * n))
        dn = x[2] * np.log(10) * 10**((x[0, :] - x[1, :] - rand) / (10 * n)) *\
            (x[0, :] - x[1, :] - rand) / 20 * n**-2
        print " ".join(("dpld:", str(dpld.shape), "dpld0:", str(dpld0.shape),
                        "dd0:", str(dd0.shape), "dn:", str(dn.shape)))
        return np.array([[dpld, dpld0, dd0], [dn]], np.float64)

    def optimize_n_least_squares(self):
        return least_squares(self.loss_least_squares, x0=0.5)

    def loss(self, d, d_hat):
        return (d - d_hat)**2

    def ldpl(self, pl, pl_0, n, d0, rnd):
        pl = pl_0 + 10 * n * np.log10(d0 / d)
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
        return (d0 * 10**((pl_d - pl_d0 - rand) / (10 * n)))

    def trilateration(self, pos_node, d_hat):
        d = sp.spatial.distance.cdist(
            pos_node[1:, :], pos_node[0, :].reshape((1, 2)))
        ''' Delete the first row '''
        r1 = d_hat[:, 0].reshape(self.n_samples, 1)
        r = r1**2 - d_hat[:, 1:]**2
        b = 0.5 * (r.transpose() + d ** 2)
        ''' Delete the first row '''
        A = pos_node[1:, :] - pos_node[0, :]
        temp = np.linalg.pinv(A).dot(b)
        temp2 = np.linalg.pinv(A).dot(b) + pos_node[0, :].reshape(2, 1)
        # print A.shape, b.shape, temp.shape, temp2.shape
        return np.linalg.pinv(A).dot(b) + pos_node[0, :].reshape(2, 1)

    def create_gaussian(self):
        x = np.linspace(-4, 4, self.n_samples)
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

    def main():
        pass

    if __name__ == "__main__":
        main()
