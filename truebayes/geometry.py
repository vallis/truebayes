import numpy as np

def setgeometry(q):
  global qdim, xmin, xmax, xstops, xmid, xwid

  # bins
  qdim = q

  # prior range for x (will be uniform)
  xmin, xmax = 0, 1

  # definition of quantization bins
  xstops = np.linspace(xmin, xmax, qdim + 1)

  # to plot histograms
  xmid = 0.5 * (xstops[:-1] + xstops[1:])
  xwid = xstops[1] - xstops[0]

setgeometry(64)