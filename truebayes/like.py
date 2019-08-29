import torch

from truebayes.utils import numpy2cuda, cuda2numpy


def synlike(a, syntrain, iterations=1000000):
  with torch.no_grad():
    # note aimag defined with plus here, exp below modified consistently
    # [areal] = [aimag] = nbasis x nsignals 
    areal, aimag = torch.t(a[:,0::2]), torch.t(a[:,1::2])

    # [anorm] = nsignals
    anorm = torch.sum(areal*areal + aimag*aimag, dim=0)
    
    cnt, norm, like = 0, 0, 0
    adapt = None
    while cnt < iterations:
      # [pxt] = nbatch x qdim
      _, pxt, alpha = syntrain()
      cnt = cnt + alpha.shape[0]

      # cpxt = torch.from_numpy(pxt).cuda()
      # handle 2D indicator array (assume square qdim)
      cpxt = numpy2cuda(pxt if pxt.ndim == 2 else pxt.reshape((pxt.shape[0], pxt.shape[1]*pxt.shape[1])),
                        alpha.dtype == torch.float32)
      
      # [alphareal] = [alphaimag] = nbatch x nbasis
      alphareal, alphaimag = alpha[:,0::2], alpha[:,1::2]

      # [norm] = qdim
      norm += torch.sum(cpxt, dim=0)
      
      # [alphanorm] = nbatch
      alphanorm = torch.sum(alphareal*alphareal + alphaimag*alphaimag, dim=1)

      # automatic normalization of exponentials based on the first batch
      loglike = alphareal @ areal + alphaimag @ aimag - 0.5*alphanorm.unsqueeze(1) - 0.5*anorm
      if adapt is None:
        adapt = torch.max(loglike, dim=0)[0]
      loglike -= adapt

      # [like] = qdim x nsignals = (qdim x nbatch) @ [(nbatch x nbasis) @ (nbasis x nsignals) + (nbatch x 1) + nsignals]
      # remember broadcasting tries to match the last dimension [so A @ b = sum(A * b,axis=1)]
      like += torch.t(cpxt) @ torch.exp(loglike)

    # (qdim x nsignals) * (qdim x 1)
    like = like / norm.unsqueeze(1)  
    
    # [ret] = nsignals x qdim = (nsignals x qdim) * qdim
    ret = torch.t(like / torch.sum(like, dim=0))
    
    nret = ret.detach().cpu().numpy()
    
  return nret if pxt.ndim == 2 else nret.reshape((nret.shape[0], pxt.shape[1], pxt.shape[1]))


def synmean(a, syntrain, iterations=1000000):
  with torch.no_grad():
    # note aimag defined with plus here, exp below modified consistently
    # [areal] = [aimag] = nbasis x nsignals 
    areal, aimag = torch.t(a[:,0::2]), torch.t(a[:,1::2])

    # [anorm] = nsignals
    anorm = torch.sum(areal*areal + aimag*aimag, dim=0)
    
    cnt = 0
    mean, square, cov, norm = 0.0, 0.0, 0.0, 0.0
    adapt = None
    while cnt < iterations:
      # [x] = nbatch
      x, _, alpha = syntrain()
      cnt = cnt + alpha.shape[0]

      x = numpy2cuda(x, alpha.dtype == torch.float32)
              
      # [alphareal] = [alphaimag] = nbatch x nbasis
      alphareal, alphaimag = alpha[:,0::2], alpha[:,1::2]
      
      # [alphanorm] = nbatch
      alphanorm = torch.sum(alphareal*alphareal + alphaimag*alphaimag, dim=1)

      # [like] = nbatch x nsignals = [(nbatch x nbasis) @ (nbasis x nsignals) + (nbatch x 1) + nsignals]
      # like = alphareal @ areal + alphaimag @ aimag - 0.5*alphanorm.unsqueeze(1) - 0.5*anorm
      like = alphareal @ areal
      like += alphaimag @ aimag
      like -= 0.5*alphanorm.unsqueeze(1)
      like -= 0.5*anorm

      if adapt is None:
        adapt = torch.max(like, dim=0)[0]
      
      like -= adapt
      like = torch.exp_(like) # in-place operation

      # [mean] = nsignals = nbatch @ (nbatch x nsignals)
      # in the 2D case, 2 x nsignals = 2 x nbatch @ (nbatch x nsignals)
      mean += torch.t(x) @ like
      square += (torch.t(x)**2) @ like
      
      if x.dim() == 2:
        cov += (x[:,0] * x[:,1]) @ like

      # [norm] = nsignals
      norm += torch.sum(like, dim=0)

      # note this tensor is very large (iter x len(a)); we should get rid of it asap        
      del like
    
    # naive variance algorithm, hope for best, see also
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    
    # [ret] = nsignals
    retmean = mean / norm
    reterr = torch.sqrt(square / norm - retmean * retmean)
    
    if x.dim() == 1:
      return cuda2numpy(retmean).T, cuda2numpy(reterr).T
    else:
      retcov = cov / norm - retmean[0,:]*retmean[1,:]
      return cuda2numpy(retmean).T, cuda2numpy(reterr).T, cuda2numpy(retcov)