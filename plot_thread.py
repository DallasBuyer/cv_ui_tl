# -*- coding: utf-8 -*-
from PyQt4 import QtCore
import matplotlib.pyplot as plt

class PlotThread(QtCore.QThread):
    finish_signal = QtCore.pyqtSignal(list)
    def __init__(self, ls, parent=None):
        super(PlotThread, self).__init__(parent)
        self.ls = ls
        self.x_len = len(self.ls[0])
        self.x = range(self.x_len)

    def run(self):
        fig = plt.figure()
        figure1 = fig.add_subplot(1, 2, 1)
        figure2 = fig.add_subplot(1, 2, 2)
        figure1.plot(self.x, self.ls[0], self.x, self.ls[2])
        figure2.plot(self.x, self.ls[1], self.x, self.ls[3])
        plt.show()