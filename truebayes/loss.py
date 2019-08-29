import math

import numpy as np
import torch

from truebayes.geometry import qdim
from truebayes.utils import numpy2cuda, cuda2numpy

def lossfunction(o, l: 'indicator'):
  """MSE loss for DNN histogram output, labels represented as indicator arrays."""

  return torch.mean(torch.sum(o**2,dim=1) - 2*torch.sum(o*l,dim=1))


def kllossfunction(o, l: 'indicator'):
  """KL loss for DNN histogram output, labels represented as indicator arrays."""

  return -torch.mean(2*torch.sum(torch.log(o)*l, dim=1))


def lossG1(o, l: 'xtrue'):
  """MSE loss for normal-PDF output (represented as a mean/variance pair)."""

  # since int N^2(x;x0,s) dx = 1/(2 sqrt(pi) s)
  # the sqerr loss is 1/(2 sqrt(pi) s) - 2 * e^{-(x_tr - x0)^2/2 s^2} / sqrt(2 pi s^2)
  # multiplying by 2 sqrt(pi)
  
  return torch.mean((1 - 2*math.sqrt(2)*torch.exp(-0.5*(l - o[:,0])**2/o[:,1]**2)) / o[:,1])


def kllossGn(o, l: 'xtrue'):
  """KL loss for Gaussian-mixture output (represented as a vector of concatenated mean/variance/weight triples)."""

  x0 = o[:,0::3]
  std = o[:,1::3]
  weight = torch.softmax(o[:,2::3], dim=1)

  # numerically unstable
  # return -torch.mean(2*torch.log(torch.sum(weight * torch.exp(-0.5*(x0 - l[:,np.newaxis])**2/std**2) / torch.sqrt(2 * math.pi * std**2),dim=1)))
  
  return -torch.mean(torch.logsumexp(torch.log(weight) - 0.5*(x0 - l[:,np.newaxis])**2/std**2 - 0.5*torch.log(2 * math.pi * std**2), dim=1))


def netmeanGn(inputs, net=None, single=True):
  if isinstance(inputs, np.ndarray):
    inputs = numpy2cuda(inputs, single)
    
  pars = cuda2numpy(net(inputs))

  dx  = pars[:,0::3] 
  std = pars[:,1::3]
  pweight = torch.softmax(torch.from_numpy(pars[:,2::3]),dim=1).numpy()

  # see https://en.wikipedia.org/wiki/Mixture_distribution
  xmean = np.sum(pweight * dx, axis=1)
  xerr  = np.sqrt(np.sum(pweight * (dx**2 + std**2), axis=1) - xmean**2)

  return xmean, xerr


def kllossfunction2(o, l: 'indicator'):
  """KL loss over 2-D histogram."""

  q = o.reshape((o.shape[0], qdim, qdim))

  return torch.mean(-torch.sum(torch.log(q)*l, dim=[1,2]))


def kllossGn2(o, l: 'xtrue'):
  """KL loss for Gaussian-mixture output, 2D, precision-matrix parameters."""

  dx  = o[:,0::6] - l[:,0,np.newaxis]
  dy  = o[:,2::6] - l[:,1,np.newaxis]
  
  # precision matrix is positive definite, so has positive diagonal terms
  Fxx = o[:,1::6]**2
  Fyy = o[:,3::6]**2
  
  # precision matrix is positive definite, so has positive 
  Fxy = torch.atan(o[:,4::6]) / (0.5*math.pi) * o[:,1::6] * o[:,3::6]
  
  weight = torch.softmax(o[:,5::6], dim=1)
   
  # omitting the sqrt(4*math*pi) since it's common to all templates
  return -torch.mean(torch.logsumexp(torch.log(weight) - 0.5*(Fxx*dx*dx + Fyy*dy*dy + 2*Fxy*dx*dy) + 0.5*torch.log(Fxx*Fyy - Fxy*Fxy), dim=1))


def netmeanGn2(inputs, net=None, single=True):
  if isinstance(inputs, np.ndarray):
    inputs = numpy2cuda(inputs, single)
    
  pars = cuda2numpy(net(inputs))

  dx, dy = pars[:,0::6], pars[:,2::6] 
  
  Fxx, Fyy = pars[:,1::6]**2, pars[:,3::6]**2
  Fxy = np.arctan(pars[:,4::6]) / (0.5*math.pi) * pars[:,1::6] * pars[:,3::6]

  det = Fxx*Fyy - Fxy*Fxy
  Cxx, Cyy, Cxy = Fyy/det, Fxx/det, -Fxy/det

  pweight = torch.softmax(torch.from_numpy(pars[:,5::6]),dim=1).numpy()

  xmean, ymean = np.sum(pweight * dx, axis=1), np.sum(pweight * dy, axis=1)
  xerr,  yerr  = np.sqrt(np.sum(pweight * (dx**2 + Cxx), axis=1) - xmean**2), np.sqrt(np.sum(pweight * (dy**2 + Cyy), axis=1) - ymean**2) 
  xycov        = np.sum(pweight * (dx*dy + Cxy), axis=1) - xmean*ymean

  return np.vstack((xmean, ymean)).T, np.vstack((xerr, yerr)).T, xycov


def sqerr(o, l: 'xtrue'):
  """Squared error loss for estimator output."""

  return torch.mean((o - l)**2)