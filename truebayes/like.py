import torch

from truebayes.utils import numpy2cuda

def synlike(a, syntrain, iters=100000):
  with torch.no_grad():
    # note aimag defined with plus here, exp below modified consistently
    # [areal] = [aimag] = nbasis x nsignals 
    areal, aimag = torch.t(a[:,0::2]), torch.t(a[:,1::2])

    # [anorm] = nsignals
    anorm = torch.sum(areal*areal + aimag*aimag, dim=0)
    
    cnt, norm, like = 0, 0, 0
    while cnt < iters:
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

      # [like] = qdim x nsignals = (qdim x nbatch) @ [(nbatch x nbasis) @ (nbasis x nsignals) + (nbatch x 1) + nsignals]
      # remember broadcasting tries to match the last dimension [so A @ b = sum(A * b,axis=1)]
      like += torch.t(cpxt) @ torch.exp(alphareal @ areal + alphaimag @ aimag - 0.5*alphanorm.unsqueeze(1)) # - 0.5*anorm)

    # (qdim x nsignals) * (qdim x 1)
    like = like / norm.unsqueeze(1)  
    
    # [ret] = nsignals x qdim = (nsignals x qdim) * qdim
    ret = torch.t(like / torch.sum(like, dim=0))
    
    nret = ret.detach().cpu().numpy()
    
  return nret if pxt.ndim == 2 else nret.reshape((nret.shape[0], pxt.shape[1], pxt.shape[1]))