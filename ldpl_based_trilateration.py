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

from ldpl import ldpl_based_trilateration

def main():
    # ldpl_model = ldpl_based_trilateration()
    print dir(ldpl_based_trilateration)

if __name__ == "__main__":
    main()
