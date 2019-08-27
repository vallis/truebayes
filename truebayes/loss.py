import torch

# common number of quantization bins
qdim = 32


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


def kllossfunction2(o, l: 'indicator'):
  """KL loss over 2-D histogram."""

  q = o.reshape((o.shape[0], qdim, qdim))

  return torch.mean(-torch.sum(torch.log(q)*l, dim=[1,2]))


def kllossGn2(o, l: 'xtrue'):
  """KL loss for Gaussian-mixture output, 2D, no covariance."""

  x0 = o[:,0::5]
  xstd = o[:,1::5]
  
  y0 = o[:,2::5]
  ystd = o[:,3::5]
  
  weight = torch.softmax(o[:,4::5], dim=1)
  
  return -torch.mean(torch.logsumexp(torch.log(weight) - 0.5*(x0 - l[:,0,np.newaxis])**2/xstd**2 - 0.5*torch.log(2 * math.pi * xstd**2)
                                                       - 0.5*(y0 - l[:,1,np.newaxis])**2/ystd**2 - 0.5*torch.log(2 * math.pi * ystd**2), dim=1))


def kllossGn2cov(o, l: 'xtrue'):
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


def sqerr(o, l: 'xtrue'):
  """Squared error loss for estimator output."""

  return torch.mean((o - l)**2)
