import PyQt6
from PyQt6 import QtGui
import pyqtgraph as pg
import numpy as np
import time

win = pg.GraphicsWindow()

p1 = win.addPlot(row=0, col=0)
p2 = win.addPlot(row=1, col=0)

curve1 = p1.plot()
dot1 = p1.plot(pen=None, symbol="o")
curve2 = p2.plot()
a=np.arange()

b = np.sin(a/50)

while True:
    curve1.setData(x=a, y=b)
    dot1.setData(x=[100], y=[0.5])
    curve2.setData(x=a, y=b)
    QtGui.QGuiApplication.processEvents()




