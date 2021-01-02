#!/usr/bin/env python3
"""
Quick check of VPIC output
"""
from __future__ import print_function

import argparse
import collections
import itertools
import json
import math
import multiprocessing
import os
import random
import sys

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

mpl.use('Qt5Agg')

tindex = 0

# def show():
#     var = combo.currentText()
#     # fname = "./field_hdf5/T." + str(tindex) + "/fields_" + str(tindex) + ".h5"
#     fname = "./data_smooth/fields_" + str(tindex) + ".h5"
#     with h5py.File(fname, 'r') as fh:
#         group = fh["Timestep_" + str(tindex)]
#         dset = group[var]
#         var_name = var[1:]
#         fdata = dset[:, 0, :]
#     fig = plt.figure(figsize=[5, 8])
#     rect = [0.14, 0.75, 0.73, 0.2]
#     ax = fig.add_axes(rect)
#     cmap = plt.cm.seismic
#     dmax = 0.35
#     dmin = -dmax
#     im1 = ax.imshow(fdata.T,
#                     # extent=[xmin, xmax, zmin, zmax],
#                     # vmin=dmin, vmax=dmax,
#                     cmap=cmap, aspect='auto',
#                     origin='lower', interpolation='none')
#     plt.show()

# app = QApplication(sys.argv)
# win = QMainWindow()
# win.setGeometry(400,400,300,300)
# win.setWindowTitle("VPIC Quick Check")

# combo = QtWidgets.QComboBox(win)
# combo.addItems(["cbx", "cby", "cbz"])
# combo.move(100,100)

# button = QtWidgets.QPushButton(win)
# button.setText("Submit")
# button.clicked.connect(show)
# button.move(100,200)

# win.show()
# sys.exit(app.exec_())

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.setCentralWidget(self.canvas)

        n_data = 50
        self.xdata = list(range(n_data))
        self.ydata = [random.randint(0, 10) for i in range(n_data)]

        # We need to store a reference to the plotted line 
        # somewhere, so we can apply the new data to it.
        self._plot_ref = None
        self.update_plot()

        self.show()

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        # Drop off the first y element, append a new one.
        self.ydata = self.ydata[1:] + [random.randint(0, 10)]

        # Note: we no longer need to clear the axis.       
        if self._plot_ref is None:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            plot_refs = self.canvas.axes.plot(self.xdata, self.ydata, 'r')
            self._plot_ref = plot_refs[0]
        else:
            # We have a reference, we can use it to update the data for that line.
            self._plot_ref.set_ydata(self.ydata)

        # Trigger the canvas to update and redraw.
        self.canvas.draw()

app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
sys.exit(app.exec_())
