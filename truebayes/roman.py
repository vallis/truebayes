import os

import numpy as np
import torch

import truebayes
from truebayes.network import makenet
from truebayes.utils import numpy2cuda

# load the standard ROMAN network

layers = [4,8,16,32,64,128] + [256] * 20 + [241]

alvin = makenet(layers, softmax=False)

ar, ai = alvin(), alvin()

ar.load_state_dict(torch.load(os.path.join(truebayes.__path__[0], 'data/4d-network/ar-state.pt')))
ai.load_state_dict(torch.load(os.path.join(truebayes.__path__[0], 'data/4d-network/ai-state.pt')))

ar.eval()
ai.eval()


def syntrain_snr(snr=[8,12], size=100000, varx='Mc', nets=None, seed=None, noise=1, varall=False,
             region=[[0.2,0.5],[0.2,0.25],[-1,1],[-1,1]], single=True):
  """Makes a training set using the ROMAN NN. It returns labels (for `varx`,
  or for all if `varall=True`), indicator vectors, and ROM coefficients
  (with `snr` and `noise`). Note that the coefficients are kept on the GPU.
  Parameters are sampled randomly within `region`."""

  if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)

  with torch.no_grad():
    xs = torch.zeros((size,4),dtype=torch.float,device='cuda:0')
    
    for i,r in enumerate(region):
      xs[:,i] = r[0] + (r[1] - r[0])*torch.rand((size,), dtype=torch.float, device='cuda:0')
    
    # handle banks with reduced dimensionality 
    for i in range(len(region),4):
      xs[:,i] = 0.0

    snrs = numpy2cuda(np.random.uniform(*snr,size=size))
      
    alphas = torch.zeros((size, 241*2), dtype=torch.float if single else torch.double, device='cuda:0')

    alphar, alphai = nets[0](xs), nets[1](xs)
    norm = torch.sqrt(torch.sum(alphar*alphar + alphai*alphai,dim=1))
 
    alphas[:,0::2] = snrs[:,np.newaxis] * alphar / norm[:,np.newaxis] + noise * torch.randn((size,241), device='cuda:0')
    alphas[:,1::2] = snrs[:,np.newaxis] * alphai / norm[:,np.newaxis] + noise * torch.randn((size,241), device='cuda:0')
  
  xr = np.zeros((size, 5),'d')
  xr[:,:4] = xs.detach().cpu().double().numpy()
  xr[:,4] = snrs.detach().cpu()
  
  del xs, alphar, alphai, norm

  # normalize (for provided regions)
  for i, r in enumerate(region):
    xr[:,i] = (xr[:,i] - r[0]) / (r[1] - r[0])

  if isinstance(varx, list):
    ix = ['Mc','nu','chi1','chi2'].index(varx[0])
    jx = ['Mc','nu','chi1','chi2'].index(varx[1])    

    i = np.digitize(xr[:,ix], xstops, False) - 1
    i[i == -1] = 0; i[i == qdim] = qdim - 1
    px = np.zeros((size, qdim), 'd'); px[range(size), i] = 1

    j = np.digitize(xr[:,jx], xstops, False) - 1
    j[j == -1] = 0; j[j == qdim] = qdim - 1
    py = np.zeros((size, qdim), 'd'); py[range(size), j] = 1

    if varall:
      return xr, np.einsum('ij,ik->ijk', px, py), alphas
    else:
      return xr[:,[ix,jx]], np.einsum('ij,ik->ijk', px, py), alphas    
  else:
    ix = ['Mc','nu','chi1','chi2'].index(varx)
  
    i = np.digitize(xr[:,ix], xstops, False) - 1
    i[i == -1] = 0; i[i == qdim] = qdim - 1
    px = np.zeros((size, qdim), 'd'); px[range(size), i] = 1
  
    if varall:
      return xr, px, alphas
    else:
      return xr[:,ix], px, alphas


def syntrainer(net, syntrain, lossfunction=lossfunction,
               batchsize=25000, iterations=300, initstep=1e-3, finalv=1e-5, clipgradient=None, validation=None, single=True):
  """Trains network NN against training sets obtained from `syntrain`,
  iterating at most `iterations`; stops if the derivative of loss
  (averaged over 20 epochs) becomes less than `finalv`."""

  indicatorloss = 'l' in lossfunction.__annotations__ and lossfunction.__annotations__['l'] == 'indicator'  
  
  if validation is not None:
    raise NotImplementedError
    
    vlabels = numpy2cuda(validation[1] if indicatorloss else validation[0], single)
    vinputs = numpy2cuda(validation[2], single)
  
  optimizer = optim.Adam(net.parameters(), lr=initstep)

  training_loss, validation_loss = [], []
  
  for epoch in range(iterations):
    t0 = time.time()

    xtrue, indicator, inputs = syntrain()
    labels = numpy2cuda(indicator if indicatorloss else xtrue, single)

    batches = inputs.shape[0]//batchsize
    averaged_loss = 0.0    
    
    for i in range(batches):
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs[i*batchsize:(i+1)*batchsize])
      loss = lossfunction(outputs, labels[i*batchsize:(i+1)*batchsize])
      loss.backward()
      
      if clipgradient is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), clipgradient)
      
      optimizer.step()

      # print statistics
      averaged_loss += loss.item()

    training_loss.append(averaged_loss/batches)

    if validation is not None:
      loss = lossfunction(net(vinputs), vlabels)
      validation_loss.append(loss.detach().cpu().item())

    if epoch == 1:
      print("One epoch = {:.1f} seconds.".format(time.time() - t0))

    if epoch % 50 == 0:
      print(epoch,training_loss[-1],validation_loss[-1] if validation is not None else '')

    try:
      if len(training_loss) > iterations/10:
        training_rate = np.polyfit(range(20), training_loss[-20:], deg=1)[0]
        if training_rate < 0 and training_rate > -finalv:
          print(f"Terminating at epoch {epoch} because training loss stopped improving sufficiently: rate = {training_rate}")
          break

      if len(validation_loss) > iterations/10:
        validation_rate = np.polyfit(range(20), validation_loss[-20:], deg=1)[0]        
        if validation_rate > 0:
          print(f"Terminating at epoch {epoch} because validation loss started worsening: rate = {validation_rate}")
          break
    except:
      pass
          
  print("Final",training_loss[-1],validation_loss[-1] if validation is not None else '')
      
  if hasattr(net,'steps'):
    net.steps += iterations
  else:
    net.steps = iterations
