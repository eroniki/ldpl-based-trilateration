#!/usr/bin/python
# coding=utf8
from __future__ import division

import numpy as np
import numpy.matlib
import scipy as sp


class residual_analysis(object):
    """docstring for residual_analysis."""

    def __init__(self, x, x_hat, metric="euclid"):
        super(residual_analysis, self).__init__()
        self.x = x
        self.x_hat = x_hat
        self.metric = metric
        self.ndata, _, self.nsample = self.x_hat.shape
        # print self.ndata, self.nsample
        self.e_vec = self.calculate_errors(x=self.x,
                                           x_hat=self.x_hat,
                                           ndata=self.ndata,
                                           nsample=self.nsample,
                                           metric=self.metric)

        self.min_e = np.min(self.e_vec, axis=1)
        self.mean_e = np.mean(self.e_vec, axis=1)
        self.max_e = np.max(self.e_vec, axis=1)

    def calculate_errors(self, x, x_hat, ndata, nsample, metric):
        if metric == "euclidean":
            e_vec = np.zeros((self.ndata, self.nsample), np.float64)
            for i in range(self.ndata):
                x_ = x[i, :].reshape(2, 1)
                e_vec[i, :] = self.euc_dist(x=x_, x_hat=x_hat[i, :, :])
        else:
            e_vec = np.zeros((self.ndata, 1), np.float64)
            for i in range(self.ndata):
                x_ = x[i, :].reshape(1, 2)
                x_hat_ = x_hat[i, :, :].transpose()
                e_vec[i] = self.mah_dist(x=x_, x_hat=x_hat_)
        return e_vec

    def localization_cdf(self, errors, nbins=None):
        if nbins is None:
            bins = np.sort(errors)
            n = errors.size
            cdf = np.array(range(n)) / float(n)
            counts = None
        else:
            counts, bins = np.histogram(error, nbins)
            cdf = np.cumsum(counts) / float(np.sum(counts))
            bins = bins[1:]

        return bins, counts, cdf

    def euc_dist(self, x, x_hat):
        x = x.transpose()
        x_hat = x_hat.transpose()
        d = sp.spatial.distance.cdist(x, x_hat, metric="euclidean")
        return d

    def mah_dist(self, x, x_hat):
        # std = np.std(x_hat, axis=0, ddof=1)
        # print "std.shape", std.shape, "std", std
        # x_hat = x_hat / std
        # print "xhat shape", x_hat.shape
        mu = np.mean(x_hat, axis=0)
        # print "Mu", mu
        diff = x - mu
        sigma = np.cov(x_hat.T)
        # print sigma.shape
        # print sigma
        # SI = np.eye(2, dtype=np.float64)
        # if numpy.linalg.matrix_rank(sigma) < sigma.size / 2
        try:
            SI = np.linalg.inv(sigma)
        except np.linalg.LinAlgError as e:
            print e
            print "x_hat", repr(x_hat)
            print "sigma", repr(sigma)
            print "cond", repr(np.linalg.matrix_rank(sigma))
            print "pinv", repr(np.linalg.pinv(sigma))
            SI = np.linalg.pinv(sigma)
        return np.sqrt((diff).dot(SI).dot(diff.T))

    def x():
        doc = "The x property."

        def fget(self):
            return self._x

        def fset(self, value):
            self._x = value

        def fdel(self):
            del self._x
        return locals()
    x = property(**x())

    def x_hat():
        doc = "The x_hat property."

        def fget(self):
            return self._x_hat

        def fset(self, value):
            self._x_hat = value

        def fdel(self):
            del self._x_hat
        return locals()
    x_hat = property(**x_hat())


def main():
    pass


if __name__ == "__main__":
    main()
