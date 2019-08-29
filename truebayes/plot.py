import math
import types

import numpy as np
import matplotlib.pyplot as pp

import torch

from truebayes.geometry import xmid, xwid

def plotgauss(xtrue, indicator, inputs, net=None, like=None, varx=None, istart=0, single=True):  
  pp.figure(figsize=(12,8))

  if isinstance(like, types.FunctionType):
    like = like(inputs)

  for i in range(6):
    pp.subplot(3,2,i+1)

    # make and plot Gaussian mixture
    
    netinput = inputs[istart+i:(istart+i+1),:]
    pars = net(netinput).detach().cpu().numpy().flatten()
        
    if len(pars) == 2:
      pdf = np.exp(-0.5*(xmid - pars[0])**2/pars[1]**2) / math.sqrt(2*math.pi*pars[1]**2) * xwid
    else:
      wg = torch.softmax(torch.from_numpy(pars[2::3]), dim=0).numpy()
      pdf = np.sum(wg * np.exp(-0.5*(xmid[:,np.newaxis] - pars[0::3])**2/pars[1::3]**2) / np.sqrt(2*math.pi*pars[1::3]**2) * xwid, axis=1)
    
    pp.plot(xmid, pdf, color='C0')
    
    # plot likelihood
    pp.plot(xmid, like[istart+i], color='C1')

    # show true x
    if xtrue.ndim == 2:
      ix = ['Mc','nu','chi1','chi2'].index(varx)
      pp.axvline(xtrue[istart+i, ix], color='C2', ls=':')
    else:
      pp.axvline(xtrue[istart+i], color='C2', ls=':')