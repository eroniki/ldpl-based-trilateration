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
        
def main():
    pass

if __name__ == "__main__":
    main()
