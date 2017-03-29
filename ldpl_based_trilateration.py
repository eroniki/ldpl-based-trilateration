#!/usr/bin/python
from __future__ import division

import numpy as np
import numpy.matlib
import scipy as sp
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ldpl import ldpl_based_trilateration
from ldpl import grid_cells
from ldpl import residual_analysis
from ldpl import visualization_tool

from wifi_data import wifi_data as db

def print_dataset_shapes(dataset):
    print "WiFi Set: ", dataset.wifi_train.shape, dataset.wifi_test.shape
    print "BT Set: ", dataset.bt_train.shape, dataset.bt_test.shape
    print "LoRa Set: ", dataset.lora_train.shape, dataset.lora_test.shape

    print "Grid Labels (by XY): ", dataset.grid_labels_train.shape, dataset.grid_labels_test.shape
    print "Grid Labels (by #): ", dataset.grid_numbers_train.shape, dataset.grid_numbers_test.shape
    print "Min and Max X Label: ", np.amin(dataset.__grid_labels_by_xy__[:,0]), np.amax(dataset.__grid_labels_by_xy__[:,0])
    print "Min and Max Y Label: ", np.amin(dataset.__grid_labels_by_xy__[:,1]), np.amax(dataset.__grid_labels_by_xy__[:,1])

def main():
    fname_dataset = "hancock_data.mat"
    fname_model = "grid_classifier.h5"
    folderLocation = os.path.dirname(os.path.realpath(__file__))
    # Create the dataset
    dataset = db(folder_location=folderLocation, filename=fname_dataset, normalize=False, missingValues=0, nTraining=1200, nTesting=343, nValidation=0, verbose=False)

    print_dataset_shapes(dataset)
    ''' pos_node_vis is just used for visualization purposes, not used in any calculations '''
    pos_node_vis = np.array([[0, 0], [8, 0], [16, 0], [25, 0], [25, 8], [16, 9], [8, 9], [0, 9]])
    ref_grids = np.array([[0, 2], [8, 2], [16, 2], [24, 2], [24, 6], [16, 6], [8, 6], [0, 6]])
    pos_node = np.array([[0, 0], [7.2000, 0], [14.4000, 0], [22.5000, 0], [22.5000, 7.2000], [14.4000, 7.2000], [7.2000, 7.2000], [0, 7.2000]]);
    ldpl_wifi = ldpl_based_trilateration(measurement=dataset.__wifi_data__, grid_labels=dataset.__grid_labels_by_xy__, pt=15, pos_node=pos_node)

    wifi_grids = grid_cells()
    ldpl_wifi.mean_measurements = wifi_grids.mean_measurements(dataset.__wifi_data__, dataset.__grid_labels_by_xy__)
    bt_grids = grid_cells()
    prop_bt = bt_grids.mean_measurements(dataset.__bt_data__, dataset.__grid_labels_by_xy__)
    lora_grids = grid_cells()
    prop_lora = lora_grids.mean_measurements(dataset.__lora_data__, dataset.__grid_labels_by_xy__)

    res_analysis = residual_analysis(error="euclidean")

if __name__ == "__main__":
    main()
