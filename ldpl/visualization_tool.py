#!/usr/bin/python
# coding=utf8
'''
    ldpl.visualization_tool

'''
from __future__ import division

import numpy as np
from random import randint
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from mpl_toolkits.mplot3d import Axes3D


class visualization_tool(object):
    """
    This class implements necessary methods to visualize an environment in
    which an agent is localized with radiolocation.
    """

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
        plt.ion()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        self.n_grid_x = n_grid_x
        self.n_grid_y = n_grid_y
        self.grid_size = grid_size
        self.colors = ((0.25882353,  0.5254902,  0.95686275),
                       (0.25490196,  0.95686275,  0.70980392),
                       (0.72156863,  0.23921569,  0.92941176),
                       (0.72156863,  0.53921569,  0.92941176),
                       (0.64705882,  0.01176471,  0.03137255),
                       (1.,  0.54901961,  0.),
                       (0.0627451,  0.76470588,  0.9372549),
                       (1.,  0.54901961,  0.))

        self.fig, self.ax = self.create_empty_canvas()
        self.mark_env(x=x, y=y, width=width, height=height)
        self.p = self.mark_anchor_nodes(pos_node=pos_node)

    def plot_loss(self, n, loss):
        _ = self.ax.plot(n, loss)

    def create_empty_canvas(self):
        fig = plt.figure(figsize=(19, 16))
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

    def mark_groundtruth(self, pos):
        p = self.mark_position(pos=pos,
                               marker='*',
                               markersize=10,
                               color="green",
                               fillstyle="full",
                               label="Ground Truth")
        return p

    def mark_position_estimation(self, pos):
        s = self.ax.scatter(pos[0, :],
                            pos[1, :],
                            s=100,
                            color="red",
                            marker='+',
                            label="Estimated Point")
        return s

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

        for i in range(n_node):
            fs = fillstyles[randint(0, len(fillstyles) - 2)]
            label = " ".join(("Anchor Node", str(i)))
            p = self.mark_position(
                pos=pos_node[i, :],
                marker="o",
                color=self.colors[i],
                markersize=10,
                fillstyle=fs,
                label=label)
        return p

    def plot_radial_distance(self, center, r):
        n_node, n_est = r.shape
        print "n_est:", n_est
        for i in range(n_node):
            norm = r[i, :] / float(np.sum(r[i, :]))
            alpha = 0.2 * norm + 0.6
            # print alpha, alpha.shape
            for j in range(n_est):
                # print "node: ", i, " est: ", j, "o: ", center[i, :], " r: ",
                # r[i, j], " c: ", self.colors[i], " a: ", np.minimum(alpha[j]
                # * 1, 1), '\n'
                label = "_"
                if i == 0 and j == 0:
                    label = "Radial Estimation"

                self.ax.add_patch(
                    patches.Circle(
                        center[i, :],
                        radius=r[i, j],
                        color=self.colors[i],
                        alpha=np.minimum(alpha[j] * 1, 1),
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
                color='black',
                alpha=0.1,
                hatch="/",
                label="Localization Environment"
            )
        )

    def mark_position(self,
                      pos,
                      marker,
                      label,
                      color,
                      markersize=10,
                      fillstyle='full'):

        p, = self.ax.plot(pos[0],
                          pos[1],
                          marker=marker,
                          color=color,
                          markersize=markersize,
                          fillstyle=fillstyle,
                          label=label,)

        return p

    def show_plots(self, fig):
        # plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.flush_events()
        # plt.pause(0.05)

    def clear_canvas(self, labels):
        children = self.ax.get_children()
        for i in range(len(children)):
            for j in range(len(labels)):
                if(children[i].get_label() == labels[j]):
                    children[i].remove()

    def set_title(self, title):
        self.ax.set_title(title)

    def show_legend(self):
        """Enable legend on axis ax"""
        self.ax.legend(bbox_to_anchor=(1, 1), loc="upper right",
                       numpoints=1, scatterpoints=1,
                       shadow=True, ncol=2)

    def save_figure(self, fig=None, folder="output", fname='plot.png', i=None):
        if fig is None:
            fig = self.fig

        if i is not None:
            fullname = "".join((folder, "/", "plot_", str(i)))
        else:
            fullname = "".join((folder, "/", fname))
        try:
            fig.savefig(fullname, dpi=fig.dpi, bbox_inches='tight')
        except IOError:
            os.makedirs(folder)
            fig.savefig(fullname, dpi=fig.dpi, bbox_inches='tight')


def main():
    pass


if __name__ == "__main__":
    main()
