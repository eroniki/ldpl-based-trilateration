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

def lora(dataset, pos_node_vis, ref_grids, pos_node):
    ''' The grid objects representing the measurements and grid centers '''
    grids = grid_cells()
    ''' Mean measurements collected at the center of the grid cell '''
    mm = grids.mean_measurements(dataset.__lora_data__, dataset.__grid_labels_by_xy__)
    ''' The reference grid centers (in order to calculated d0 and PL_d0) '''
    pr_ref = mm[ref_grids[:,0], ref_grids[:,1], np.arange(8)]
    ''' The center of the grid cells '''
    center_ref = grids.centers[ref_grids[:,0], ref_grids[:,1]]
    d0 = sp.spatial.distance.cdist(center_ref, pos_node).diagonal()
    ''' The distance between the measurement point and the anchor node '''
    qq = np.vstack((grids.centers_x[dataset.__grid_labels_by_xy__[:,0]], \
        grids.centers_y[dataset.__grid_labels_by_xy__[:,1]])).transpose()
    d = sp.spatial.distance.cdist(qq, pos_node)
    ''' Construct the ldpl model for wifi '''
    ldpl = ldpl_based_trilateration(measurement=dataset.__wifi_data__, \
        grid_labels=dataset.__grid_labels_by_xy__, pt=24, pos_node=pos_node,\
        pr_d0= pr_ref, d0=d0, d=d)
    ldpl.mean_measurements = mm
    n_test = np.linspace(0.5, 15, 200)
    # loss = ldpl.optimize_n(n_test=n_test)
    loss = 0
    popt, pcov = ldpl.curve_fit()
    print popt, pcov
    ''' Print the optimal path loss exponents '''
    print " ".join(("min loss:", str(np.min(loss)), "argmin_loss:", str(n_test[np.argmin(loss)])))
    # r = ldpl.ldpl(ldpl.pl, ldpl.pl_d0, popt, ldpl.d0, ldpl.rand)
    d_hat = np.zeros((1543, 1000, 8))
    for i in range(ldpl.n_data):
        d_hat[i, :, :] = ldpl.get_radial_distance(ldpl.pl[i,:], ldpl.pl_d0, popt, ldpl.d0)
    print d_hat[0,:,:].shape
    x_hat = np.zeros((1543,2, 1000))
    for i in range(1543):
        x_hat[i,:, :] = ldpl.trilateration(pos_node, d_hat[i,:,:])
        gt = np.vstack((grids.centers_x[dataset.__grid_labels_by_xy__[i,0]], \
            grids.centers_y[dataset.__grid_labels_by_xy__[i,1]])).transpose()
        d = sp.spatial.distance.cdist(qq, pos_node)
        print " ".join(("x_hat:", str(np.mean(x_hat, axis=1)), "gt:", str(gt)))
    qq = np.vstack((grids.centers_x[dataset.__grid_labels_by_xy__[:,0]], \
        grids.centers_y[dataset.__grid_labels_by_xy__[:,1]])).transpose()
    e =  x_hat - qq
    # e = sp.spatial.distance.cdist(x_hat, np.vstack((grids.centers_x[dataset.__grid_labels_by_xy__[:,0]], \
    #     grids.centers_y[dataset.__grid_labels_by_xy__[:,1]])).transpose())
    print np.mean(e), np.max(e), np.min(e)
    return n_test, loss

def wifi(dataset, pos_node_vis, ref_grids, pos_node):
    ''' The grid objects representing the measurements and grid centers '''
    grids = grid_cells()
    ''' Mean measurements collected at the center of the grid cell '''
    mm = grids.mean_measurements(dataset.__wifi_data__, dataset.__grid_labels_by_xy__)
    ''' The reference grid centers (in order to calculated d0 and PL_d0) '''
    pr_ref = mm[ref_grids[:,0], ref_grids[:,1], np.arange(8)]
    ''' The center of the grid cells '''
    center_ref = grids.centers[ref_grids[:,0], ref_grids[:,1]]
    d0 = sp.spatial.distance.cdist(center_ref, pos_node).diagonal()
    ''' The distance between the measurement point and the anchor node '''
    qq = np.vstack((grids.centers_x[dataset.__grid_labels_by_xy__[:,0]], \
        grids.centers_y[dataset.__grid_labels_by_xy__[:,1]])).transpose()
    d = sp.spatial.distance.cdist(qq, pos_node)
    ''' Construct the ldpl model for wifi '''
    ldpl = ldpl_based_trilateration(measurement=dataset.__wifi_data__, \
        grid_labels=dataset.__grid_labels_by_xy__, pt=15, pos_node=pos_node,\
        pr_d0= pr_ref, d0=d0, d=d)
    ldpl.mean_measurements = mm
    n_test = np.linspace(0.5, 200, 200)
    loss = ldpl.optimize_n(n_test=n_test)
    popt, pcov = ldpl.curve_fit()
    print popt, pcov
    ''' Print the optimal path loss exponents '''
    print " ".join(("min loss:", str(np.min(loss)), "argmin_loss:", str(n_test[np.argmin(loss)])))
    return n_test, loss

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
    pos_node = np.array([[0, 0], [7.2000, 0], [14.4000, 0], [22.5000, 0], [22.5000, 7.2000], [14.4000, 7.2000], [7.2000, 7.2000], [0, 7.2000]])

    # TODO: pr_ref_bt contains, nan values; find another reference grids for bt measurements
    ''' FOR DEBUGGING PURPOSES '''
    # print wifi_grids.centers.shape
    # print "grid centers\n", wifi_grids.centers
    # print "center_ref\n", center_ref
    # print "ref grids\n", ref_grids
    # print "d0\n", d0
    n_test_lora, loss_lora = lora(dataset, pos_node_vis, ref_grids, pos_node)
    # n_test_wifi, loss_wifi = wifi(dataset, pos_node_vis, ref_grids, pos_node)
    # plt.plot(n_test_lora, loss_lora, 'b', n_test_wifi, loss_wifi, 'g')
    # plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
    # plt.ylim([350, 1000])
    # plt.minorticks_on()
    # plt.show()
    #
    # res_analysis = residual_analysis(error="euclidean")

if __name__ == "__main__":
    main()
