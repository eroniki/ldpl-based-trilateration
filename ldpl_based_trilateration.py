#!/usr/bin/python
from __future__ import division

import numpy as np
import numpy.matlib
import scipy as sp
from scipy.spatial import distance
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
    wifi_grids = grid_cells()
    bt_grids = grid_cells()
    lora_grids = grid_cells()
    fname_dataset = "hancock_data.mat"
    fname_model = "grid_classifier.h5"
    folderLocation = os.path.dirname(os.path.realpath(__file__))
    # Create the dataset
    dataset = db(folder_location=folderLocation, filename=fname_dataset, normalize=False, missingValues=0, nTraining=1200, nTesting=343, nValidation=0, verbose=False)

    print_dataset_shapes(dataset)
    ''' pos_node_vis is just used for visualization purposes, not used in any calculations '''
    pos_node_vis = np.array([[0, 0], [8, 0], [16, 0], [25, 0], [25, 8], [16, 9], [8, 9], [0, 9]])
    ref_grids = np.array([[0, 2], [8, 2], [16, 2], [24, 2], [24, 6], [16, 6], [8, 6], [0, 6]])
    pos_node = np.array([[0, 0], [7.2000, 0], [14.4000, 0], [22.5000, 0], [22.5000, 7.2000], [14.4000, 7.2000], [7.2000, 7.2000], [0, 7.2000]])

    mm_wifi = wifi_grids.mean_measurements(dataset.__wifi_data__, dataset.__grid_labels_by_xy__)
    pr_ref_wifi = mm_wifi[ref_grids[:,0], ref_grids[:,1], np.arange(8)]
    mm_bt = bt_grids.mean_measurements(dataset.__bt_data__, dataset.__grid_labels_by_xy__)
    pr_ref_bt = mm_bt[ref_grids[:,0], ref_grids[:,1], np.arange(8)]
    # TODO: pr_ref_bt contains, nan values; find another reference grids for bt measurements
    mm_lora = lora_grids.mean_measurements(dataset.__lora_data__, dataset.__grid_labels_by_xy__)
    pr_ref_lora = mm_lora[ref_grids[:,0], ref_grids[:,1], np.arange(8)]
    center_ref = wifi_grids.centers[ref_grids[:,0], ref_grids[:,1]]
    ''' FOR DEBUGGING PURPOSES '''
    # print wifi_grids.centers.shape
    # print "grid centers\n", wifi_grids.centers
    # print "center_ref\n", center_ref
    # print "ref grids\n", ref_grids
    # print "d0\n", d0
    ''' The distance between the center of the reference grids and the anchor
    node positions '''
    d0 = sp.spatial.distance.cdist(center_ref, pos_node).diagonal()
    ''' The distance between the measurement point and the anchor node '''
    qq = np.vstack((wifi_grids.centers_x[dataset.__grid_labels_by_xy__[:,0]], \
        wifi_grids.centers_y[dataset.__grid_labels_by_xy__[:,1]])).transpose()
    d = sp.spatial.distance.cdist(qq, pos_node)
    ''' Construct the ldpl model for wifi '''
    ldpl_wifi = ldpl_based_trilateration(measurement=dataset.__wifi_data__, \
        grid_labels=dataset.__grid_labels_by_xy__, pt=15, pos_node=pos_node,\
        pr_d0= pr_ref_wifi, d0=d0, d=d)
    ldpl_wifi.mean_measurements = mm_wifi
    n_test = np.linspace(0.5, 15, 200)
    loss_wifi = ldpl_wifi.optimize_n(n_test=n_test)
    # print loss_wifi
    ''' Construct the ldpl model for lora '''
    ldpl_lora = ldpl_based_trilateration(measurement=dataset.__lora_data__, \
        grid_labels=dataset.__grid_labels_by_xy__, pt=24, pos_node=pos_node,\
        pr_d0= pr_ref_lora, d0=d0, d=d)
    ldpl_lora.mean_measurements = mm_lora
    n_test = np.linspace(0.5, 15, 200)
    loss_lora = ldpl_lora.optimize_n(n_test=n_test)
    # print loss_lora
    ''' Print the optimal path loss exponents '''
    print " ".join(("min loss:", str(np.min(loss_wifi)), "argmin_loss:", str(n_test[np.argmin(loss_wifi)])))
    print " ".join(("min loss:", str(np.min(loss_lora)), "argmin_loss:", str(n_test[np.argmin(loss_lora)])))
    # # # print loss, np.min(loss), np.argmin(loss)
    plt.plot(n_test, (loss_wifi), 'r', n_test, (loss_lora), 'b')
    plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
    # plt.ylim([350, 1000])
    plt.minorticks_on()
    plt.show()

    res_analysis = residual_analysis(error="euclidean")

if __name__ == "__main__":
    main()
