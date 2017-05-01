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
from ldpl import visualization_tool as vis

from wifi_data import wifi_data as db


def print_dataset_shapes(dataset):
    print "WiFi Set: ", dataset.wifi_train.shape, dataset.wifi_test.shape
    print "BT Set: ", dataset.bt_train.shape, dataset.bt_test.shape
    print "LoRa Set: ", dataset.lora_train.shape, dataset.lora_test.shape

    print "Grid Labels (by XY): ", dataset.grid_labels_train.shape,
    dataset.grid_labels_test.shape
    print "Grid Labels (by #): ", dataset.grid_numbers_train.shape,
    dataset.grid_numbers_test.shape
    print "Min and Max X Label: ",
    np.amin(dataset.__grid_labels_by_xy__[:, 0]),
    np.amax(dataset.__grid_labels_by_xy__[:, 0])
    print "Min and Max Y Label: ",
    np.amin(dataset.__grid_labels_by_xy__[:, 1]),
    np.amax(dataset.__grid_labels_by_xy__[:, 1])


def lora(dataset, pos_node_vis, ref_grids, pos_node):
    '''The grid objects representing the measurements and grid centers'''
    grids = grid_cells()
    '''Mean measurements collected at the center of the grid cell'''
    mm = grids.mean_measurements(
        dataset.__lora_data__, dataset.__grid_labels_by_xy__)
    '''The reference grid centers (in order to calculated d0 and PL_d0)'''
    pr_ref = mm[ref_grids[:, 0], ref_grids[:, 1], np.arange(8)]
    '''The center of the grid cells'''
    center_ref = grids.centers[ref_grids[:, 0], ref_grids[:, 1]]
    d0 = sp.spatial.distance.cdist(center_ref, pos_node).diagonal()
    '''The distance between the measurement point and the anchor node'''
    qq = np.vstack((grids.centers_x[dataset.__grid_labels_by_xy__[:, 0]],
                    grids.centers_y[dataset.__grid_labels_by_xy__[:, 1]]
                    )).transpose()
    d = sp.spatial.distance.cdist(qq, pos_node)
    '''Construct the ldpl model for wifi'''
    ldpl = ldpl_based_trilateration(measurement=dataset.__wifi_data__,
                                    grid_labels=dataset.__grid_labels_by_xy__,
                                    pt=24,
                                    pos_node=pos_node,
                                    pr_d0=pr_ref, d0=d0, d=d)
    ldpl.mean_measurements = mm
    n_test = np.linspace(0.5, 15, 200)
    '''self.optimize_n() function is not sensible right now. It should be used to
    show the shape of the loss'''
    # loss = ldpl.optimize_n(n_test=n_test)
    '''It takes some time to calculate, no need to run everytime'''
    popt, pcov = ldpl.curve_fit()
    # popt, pcov = np.array([3.26754357, 1]), None
    n, sigma = popt
    print popt, pcov
    '''Print the optimal path loss exponents'''
    # print " ".join(("min loss:", str(np.min(loss)),
    #                 "argmin_loss:",
    #                 str(n_test[np.argmin(loss)])))
    d_hat_matrice = np.zeros((ldpl.n_data, 1000, 8))
    x_hat_matrice = np.zeros((ldpl.n_data, 2, 1000))
    gt = np.vstack((grids.centers_x[dataset.__grid_labels_by_xy__[:, 0]],
                    grids.centers_y[dataset.__grid_labels_by_xy__[:, 1]]))\
        .transpose()
    '''Start estimating using the optimal path loss exponent'''
    for i in range(ldpl.n_data):
        d_hat = ldpl.get_radial_distance(
            ldpl.pl[i, :], ldpl.pl_d0, n, ldpl.d0, sigma)

        x_hat = ldpl.trilateration(pos_node, d_hat)

        d_hat_matrice[i, :, :] = d_hat
        x_hat_matrice[i, :, :] = x_hat
        # r_mean = np.mean(d_hat, axis=0)
        # r_mean = r_mean.reshape(r_mean.size, 1)
        # e = x_hat.mean(axis=1) - gt[i, :]
        # debug = " ".join(("Data #:", str(i),
        #                   "\nx_hat:", str(x_hat.mean(axis=1)),
        #                   "\ngt:", str(gt[i, :]),
        #                   "\ne:", str(e),
        #                   "\ne_2:", str(np.linalg.norm(e)),
        #                   "\nx_hat shape:", str(x_hat.shape),
        #                   "\nd_hat_shape:", str(d_hat.shape),
        #                   "\nr_mean:", str(r_mean)))
        # print debug

    return ldpl, d_hat_matrice, x_hat_matrice, gt, n, sigma


def wifi(dataset, pos_node_vis, ref_grids, pos_node):
    '''The grid objects representing the measurements and grid centers'''
    grids = grid_cells()
    '''Mean measurements collected at the center of the grid cell'''
    mm = grids.mean_measurements(
        dataset.__wifi_data__, dataset.__grid_labels_by_xy__)
    '''The reference grid centers (in order to calculated d0 and PL_d0)'''
    pr_ref = mm[ref_grids[:, 0], ref_grids[:, 1], np.arange(8)]
    '''The center of the grid cells'''
    center_ref = grids.centers[ref_grids[:, 0], ref_grids[:, 1]]
    d0 = sp.spatial.distance.cdist(center_ref, pos_node).diagonal()
    '''The distance between the measurement point and the anchor node'''
    qq = np.vstack((grids.centers_x[dataset.__grid_labels_by_xy__[:, 0]],
                    grids.centers_y[dataset.__grid_labels_by_xy__[:, 1]]
                    )).transpose()
    d = sp.spatial.distance.cdist(qq, pos_node)
    '''Construct the ldpl model for wifi'''
    ldpl = ldpl_based_trilateration(measurement=dataset.__wifi_data__,
                                    grid_labels=dataset.__grid_labels_by_xy__,
                                    pt=24,
                                    pos_node=pos_node,
                                    pr_d0=pr_ref, d0=d0, d=d)
    ldpl.mean_measurements = mm
    n_test = np.linspace(0.5, 15, 200)
    '''self.optimize_n() function is not sensible right now. It should be used to
    show the shape of the loss'''
    # loss = ldpl.optimize_n(n_test=n_test)
    '''It takes some time to calculate, no need to run everytime'''
    popt, pcov = ldpl.curve_fit()
    # popt, pcov = np.array([3.26754357, 1]), None
    n, sigma = popt
    print popt, pcov
    '''Print the optimal path loss exponents'''
    # print " ".join(("min loss:", str(np.min(loss)),
    #                 "argmin_loss:",
    #                 str(n_test[np.argmin(loss)])))
    d_hat_matrice = np.zeros((ldpl.n_data, 1000, 8))
    x_hat_matrice = np.zeros((ldpl.n_data, 2, 1000))
    gt = np.vstack((grids.centers_x[dataset.__grid_labels_by_xy__[:, 0]],
                    grids.centers_y[dataset.__grid_labels_by_xy__[:, 1]]))\
        .transpose()
    '''Start estimating using the optimal path loss exponent'''
    for i in range(ldpl.n_data):
        d_hat = ldpl.get_radial_distance(
            ldpl.pl[i, :], ldpl.pl_d0, n, ldpl.d0, sigma)

        x_hat = ldpl.trilateration(pos_node, d_hat)

        d_hat_matrice[i, :, :] = d_hat
        x_hat_matrice[i, :, :] = x_hat
        # r_mean = np.mean(d_hat, axis=0)
        # r_mean = r_mean.reshape(r_mean.size, 1)
        # e = x_hat.mean(axis=1) - gt[i, :]
        # debug = " ".join(("Data #:", str(i),
        #                   "\nx_hat:", str(x_hat.mean(axis=1)),
        #                   "\ngt:", str(gt[i, :]),
        #                   "\ne:", str(e),
        #                   "\ne_2:", str(np.linalg.norm(e)),
        #                   "\nx_hat shape:", str(x_hat.shape),
        #                   "\nd_hat_shape:", str(d_hat.shape),
        #                   "\nr_mean:", str(r_mean)))
        # print debug

    return ldpl, d_hat_matrice, x_hat_matrice, gt, n, sigma


def bt(dataset, pos_node_vis, ref_grids, pos_node):
    '''The grid objects representing the measurements and grid centers'''
    grids = grid_cells()
    '''Mean measurements collected at the center of the grid cell'''
    mm = grids.mean_measurements(
        dataset.__bt_data__, dataset.__grid_labels_by_xy__)
    '''The reference grid centers (in order to calculated d0 and PL_d0)'''
    pr_ref = mm[ref_grids[:, 0], ref_grids[:, 1], np.arange(8)]
    print pr_ref
    '''The center of the grid cells'''
    center_ref = grids.centers[ref_grids[:, 0], ref_grids[:, 1]]
    d0 = sp.spatial.distance.cdist(center_ref, pos_node).diagonal()
    '''The distance between the measurement point and the anchor node'''
    qq = np.vstack((grids.centers_x[dataset.__grid_labels_by_xy__[:, 0]],
                    grids.centers_y[dataset.__grid_labels_by_xy__[:, 1]]
                    )).transpose()
    d = sp.spatial.distance.cdist(qq, pos_node)
    '''Construct the ldpl model for bt'''
    ldpl = ldpl_based_trilateration(measurement=dataset.__bt_data__,
                                    grid_labels=dataset.__grid_labels_by_xy__,
                                    pt=24,
                                    pos_node=pos_node,
                                    pr_d0=pr_ref, d0=d0, d=d)
    ldpl.mean_measurements = mm
    n_test = np.linspace(0.5, 15, 200)
    '''self.optimize_n() function is not sensible right now. It should be used to
    show the shape of the loss'''
    # loss = ldpl.optimize_n(n_test=n_test)
    '''It takes some time to calculate, no need to run everytime'''
    popt, pcov = ldpl.curve_fit()
    # popt, pcov = np.array([3.26754357, 1]), None
    n, sigma = popt
    print popt, pcov
    '''Print the optimal path loss exponents'''
    # print " ".join(("min loss:", str(np.min(loss)),
    #                 "argmin_loss:",
    #                 str(n_test[np.argmin(loss)])))
    d_hat_matrice = np.zeros((ldpl.n_data, 1000, 8))
    x_hat_matrice = np.zeros((ldpl.n_data, 2, 1000))
    gt = np.vstack((grids.centers_x[dataset.__grid_labels_by_xy__[:, 0]],
                    grids.centers_y[dataset.__grid_labels_by_xy__[:, 1]]))\
        .transpose()
    '''Start estimating using the optimal path loss exponent'''
    for i in range(ldpl.n_data):
        d_hat = ldpl.get_radial_distance(
            ldpl.pl[i, :], ldpl.pl_d0, n, ldpl.d0, sigma)

        x_hat = ldpl.trilateration(pos_node, d_hat)

        d_hat_matrice[i, :, :] = d_hat
        x_hat_matrice[i, :, :] = x_hat
        # r_mean = np.mean(d_hat, axis=0)
        # r_mean = r_mean.reshape(r_mean.size, 1)
        # e = x_hat.mean(axis=1) - gt[i, :]
        # debug = " ".join(("Data #:", str(i),
        #                   "\nx_hat:", str(x_hat.mean(axis=1)),
        #                   "\ngt:", str(gt[i, :]),
        #                   "\ne:", str(e),
        #                   "\ne_2:", str(np.linalg.norm(e)),
        #                   "\nx_hat shape:", str(x_hat.shape),
        #                   "\nd_hat_shape:", str(d_hat.shape),
        #                   "\nr_mean:", str(r_mean)))
        # print debug

    return ldpl, d_hat_matrice, x_hat_matrice, gt, n, sigma


def visualize(d_hat_matrice=None, x_hat_matrice=None, gt=None,
              pos_node=None, ndata=None, folder=None):

    vt = vis(x=0, y=0, width=22.5, height=7.2, pos_node=pos_node)
    '''Labels for the plots'''
    labels = ["Ground Truth", "Estimated Point", "Radial Estimation", "_"]
    '''Set plot children to None'''
    p = None
    for i in range(ndata):
        if p is not None:
            vt.clear_canvas(labels=labels)
        '''Mark groundtruth'''
        p = vt.mark_groundtruth(pos=gt[i, :])
        '''Mark Radial Estimations'''
        d_hat = d_hat_matrice[i, :, :]
        r_mean = np.mean(d_hat, axis=0)
        r_mean = r_mean.reshape(r_mean.size, 1)
        vt.plot_radial_distance(center=pos_node, r=r_mean)
        '''Mark position estimation'''
        x_hat = x_hat_matrice[i, :, :]
        p = vt.mark_position_estimation(pos=x_hat)
        e = x_hat.mean(axis=1) - gt[i, :]
        title = " ".join((r"Data \#:", str(i),
                          r"$x$:", str(gt[i, :]),
                          r"$\overline{\hat{\mathbf{x}}}$:", str(
                              x_hat.mean(axis=1)),
                          r"$\overline{e}$:", str(e),
                          r"${\| x - \overline{\hat{\mathbf{x}}}\|}_2$:", str(
                              np.linalg.norm(e)),
                          ))
        '''Show Results'''
        vt.set_title(title)
        vt.show_legend()
        # vt.show_plots(vt.fig)
        vt.save_figure(folder=folder, i=i)


def main():
    fname_dataset = "hancock_data.mat"
    fname_model = "grid_classifier.h5"
    folderLocation = os.path.dirname(os.path.realpath(__file__))
    # pos_node_vis is just used for visualization purposes,
    # not used in any calculations
    pos_node_vis = np.array([[0, 0], [8, 0], [16, 0], [25, 0],
                             [25, 8], [16, 9], [8, 9], [0, 9]])
    ref_grids = np.array([[0, 2], [8, 2], [16, 2], [24, 2], [24, 6], [16, 6],
                          [8, 6], [0, 6]])
    pos_node = np.array([[0, 0], [7.2000, 0], [14.4000, 0], [22.5000, 0],
                         [22.5000, 7.2000], [14.4000, 7.2000],
                         [7.2000, 7.2000], [0, 7.2000]])
    # Create the dataset
    dataset = db(folder_location=folderLocation,
                 filename=fname_dataset,
                 normalize=False,
                 missingValues=0,
                 nTraining=1200,
                 nTesting=343,
                 nValidation=0,
                 verbose=False)

    print_dataset_shapes(dataset)

    # TODO: pr_ref_bt contains, nan values; find another reference grids for
    # bt measurements
    # '''LORA'''
    # ldpl_lora, d_hat_lora, x_hat_lora, gt_lora, n_lora, \
    #     sigma_lora = lora(dataset,
    #                       pos_node_vis,
    #                       ref_grids,
    #                       pos_node)
    #
    # res_lora = residual_analysis(
    #     x=gt_lora, x_hat=x_hat_lora, metric="euclidean")

    '''WIFI'''
    ldpl_wifi, d_hat_wifi, x_hat_wifi, gt_wifi, n_wifi, \
        sigma_wifi = wifi(dataset,
                          pos_node_vis,
                          ref_grids,
                          pos_node)

    res_wifi = residual_analysis(
        x=gt_wifi, x_hat=x_hat_wifi, metric="euclidean")

    # '''BT'''
    # ref_grids_bt = np.array([[0, 2], [8, 2], [16, 2], [24, 2], [24, 7],
    #                          [16, 6], [8, 6], [1, 5]])
    # ldpl_bt, d_hat_bt, x_hat_bt, gt_bt, n_bt, \
    #     sigma_bt = bt(dataset,
    #                   pos_node_vis,
    #                   ref_grids_bt,
    #                   pos_node)
    #
    # res_bt = residual_analysis(
    #     x=gt_bt, x_hat=x_hat_bt, metric="euclidean")

    '''Print Rough Results'''
    # print res_lora.mean_e, np.mean(res_lora.mean_e)
    print res_wifi.mean_e, np.mean(res_wifi.mean_e)
    # print res_bt.mean_e, np.mean(res_bt.mean_e)
    # '''Visualize'''
    # visualize(d_hat_matrice=d_hat_lora, x_hat_matrice=x_hat_lora,
    #           gt=gt_lora, pos_node=pos_node,
    #           ndata=ldpl_lora.n_data, folder="lora_output")
    #
    # visualize(d_hat_matrice=d_hat_wifi, x_hat_matrice=x_hat_wifi,
    #           gt=gt_wifi, pos_node=pos_node,
    #           ndata=ldpl_wifi.n_data, folder="wifi_output")
    #
    # visualize(d_hat_matrice=d_hat_bt, x_hat_matrice=x_hat_bt,
    #           gt=gt_bt, pos_node=pos_node,
    #           ndata=ldpl_bt.n_data, folder="bt_output")


if __name__ == "__main__":
    main()
