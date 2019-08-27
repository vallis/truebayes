import torch

def numpy2cuda(array, single=True):
  array = torch.from_numpy(array)
  
  if single:
    array = array.float()
    
  if torch.cuda.is_available():
    array = array.cuda()
    
  return array


def cuda2numpy(tensor):
  return tensor.detach().cpu().numpy()
