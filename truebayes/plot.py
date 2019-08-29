import math
import types

import numpy as np
import matplotlib.pyplot as pp

import torch

from truebayes.geometry import xmid, xwid

def plotgauss(xtrue, indicator, inputs, net=None, like=None, varx=None, twodim=False, istart=0, single=True):  
  pp.figure(figsize=(12,8))

  if isinstance(like, types.FunctionType):
    like = like(inputs)

  for i in range(6):
    pp.subplot(3,2,i+1)

    # make and plot Gaussian mixture
    
    netinput = inputs[istart+i:(istart+i+1),:]
    pars = net(netinput).detach().cpu().numpy().flatten()
    
    if twodim:
      Fxx, Fyy = pars[1::6]**2, pars[3::6]**2
      Fxy = np.arctan(pars[4::6]) / (0.5*math.pi) * pars[1::6] * pars[3::6]
      weight = torch.softmax(torch.from_numpy(pars[5::6]),dim=0).numpy()

      dx  = (pars[0::6] if varx == 'nu' else pars[1::6]) - xmid[:,np.newaxis]
      Cxx = Fxx / (Fxx*Fyy - Fxy*Fxy) if varx == 'nu' else Fyy / (Fxx*Fyy - Fxy*Fxy), 

      pdf = np.sum(weight * np.exp(-0.5*dx**2/Cxx) / np.sqrt(2*math.pi*Cxx) * xwid, axis=1)

      # logmod = scs.logsumexp(np.log(weight) - 0.5*(Fxx*dx*dx + Fyy*dy*dy + 2*Fxy*dx*dy) + 0.5*np.log(Fxx*Fyy - Fxy*Fxy), axis=2)
    else:
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