#!/usr/bin/python
from __future__ import division

import numpy as np
import numpy.matlib
import scipy as sp
from random import randint
from scipy.stats import norm
from scipy.optimize import minimize_scalar, minimize, least_squares, curve_fit
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable


class visualization_tool(object):
    """This class implements necessary methods to visualize an environment in
    which an agent is localized with radiolocation."""

    def __init__(self,
                 pos_node,
                 x=0,
                 y=0,
                 width=22.5,
                 height=7.2,
                 n_grid_x=25,
                 n_grid_y=8,
                 grid_size=0.9):

        super(visualization_tool, self).__init__()
        # plt.ion()
        self.n_grid_x = n_grid_x
        self.n_grid_y = n_grid_y
        self.grid_size = grid_size
        self.fig, self.ax = self.create_canvas_2d()
        self.mark_env(x=x, y=y, width=width, height=height)
        self.ax, self.p = self.mark_anchor_nodes(pos_node=pos_node)

    def plot_loss(self, n, loss):
        _ = self.ax.plot(n, loss)

    def create_canvas_2d(self):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(self.grid_size))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(self.grid_size))
        ax.grid(b=True, which='both', color='r', linestyle='-', alpha=0.2)
        ax.set_aspect('equal', 'box-forced')
        ax.set_xlim((-1, self.n_grid_x * self.grid_size + 15))
        ax.set_ylim((-1, self.n_grid_y * self.grid_size + 1))
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        return fig, ax

    def create_canvas_1d(self, arg):
        pass

    def mark_groundtruth(self, ax, pos):
        ax, p = self.mark_position(ax=ax,
                                   pos=pos,
                                   marker='*',
                                   markersize=10,
                                   color="green",
                                   fillstyle="full",
                                   label='ground truth')
        return ax, p

    def mark_position_estimation(self, pos):
        self.mark_position(ax=self.ax,
                           pos=pos,
                           marker='+',
                           markersize=10,
                           color="red",
                           fillstyle="full",
                           label='estimated point')

    def mark_error_vector(self, gt, est):
        e = est - gt
        self.ax.arrow(gt[0],
                      gt[1],
                      e[0],
                      e[1],
                      head_width=0.05,
                      head_length=0.1,
                      length_includes_head=True,
                      fc='k',
                      ec='k')

    def mark_anchor_nodes(self, pos_node):
        (n_node, _) = pos_node.shape
        fillstyles = Line2D.fillStyles
        colors = ((0.25882353,  0.5254902,  0.95686275),
                  (0.25490196,  0.95686275,  0.70980392),
                  (0.72156863,  0.23921569,  0.92941176),
                  (0.72156863,  0.53921569,  0.92941176),
                  (0.64705882,  0.01176471,  0.03137255),
                  (1.,  0.54901961,  0.),
                  (0.0627451,  0.76470588,  0.9372549),
                  (1.,  0.54901961,  0.))

        for i in range(n_node):
            fs = fillstyles[randint(0, len(fillstyles) - 2)]
            label = " ".join(("Anchor Node", str(i)))
            ax, p = self.mark_position(ax=self.ax,
                                       pos=pos_node[i, :],
                                       marker="o",
                                       color=colors[i],
                                       markersize=10,
                                       fillstyle=fs,
                                       label=label)
        return ax, p

    def plot_radial_distance(self, center, r, label):
        self.ax.add_patch(
            patches.Circle(
                center,
                radius=r,
                color='green',
                alpha=0.75,
                label=label,
                fill=False
            )
        )

    def mark_env(self, x, y, width, height):
        self.ax.add_patch(
            patches.Rectangle(
                (x, y),
                width,
                height,
                color='red',
                alpha=0.25,
                hatch="/",
                label="Localization Environment"
            )
        )

    def mark_position(self,
                      ax,
                      pos,
                      marker,
                      label,
                      color,
                      markersize=10,
                      fillstyle='full'):

        p, = ax.plot(pos[0],
                     pos[1],
                     marker,
                     color=color,
                     markersize=markersize,
                     fillstyle=fillstyle,
                     label=label,)
        return ax, p

    def show_plots(self):
        plt.show(block=False)
        # plt.pause(0.05)

    def clear_canvas(self):
        self.fig.clf()
        # self.fig, self.ax = self.create_canvas_2d()

    def show_legend(self):
        """Enable legend on axis ax"""
        self.ax.legend(numpoints=1)

    def save_figure(self, fig, fname='plot.png'):
        fig.savefig(fname, dpi=fig.dpi)


def main():
    pass


if __name__ == "__main__":
    main()
