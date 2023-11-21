
# // Amira Abdel-Rahman 
# // (c) Massachusetts Institute of Technology 2023 

# based on https://github.com/UW-ERSL/JAXTOuNN
# import jax.numpy as jnp
import numpy as np
# from jax import jit
# import jax
# from constrained_ai.fab import *
import torch
import numpy as np
from scipy.sparse import coo_matrix
# import numpy.matlib
# import cvxopt
# import cvxopt.cholmod

def to_torch(x,device):
  return torch.tensor(x).to(device)

class FESolver:
  def __init__(self, mesh, material,device):
    self.device=device
    self.mesh = mesh
    self.material = material
    # self.objectiveHandle = jit(self.objective)
    
    # self.D0 = self.material.getD0elemMatrix(self.mesh)

    # self.target=target["name"]

    self.free=self.mesh.bc['free']
    self.fixed=self.mesh.bc["fixed"]
    self.iK,self.jK=self.mesh.nodeIdx
    self.ndof=self.mesh.ndof
    self.KE=material.getD0elemMatrix(mesh)
    self.edofMat=self.mesh.edofMat
    self.penal=2.0
    self.Emax=1.0
    self.Emin=0.00001
    self.f = self.mesh.bc['force'];
    self.numDOFPerNode=3
    self.numElems=self.mesh.numElems

  def objectivePytorch(self, density, penal=2.0): 

    def getYoungsModulus(density, penal):
      Y = (self.Emax)* torch.pow((density+0.01),penal)
      # Y = self.Emin + \
      #       (self.Emax-self.Emin)*\
      #         (density+0.01)**penal
      return Y 

    def assembleK(Y):
      K = to_torch(np.zeros((self.mesh.ndof, self.mesh.ndof)),self.device)
      sK = torch.flatten(torch.einsum('e, jk->ejk', Y, to_torch(self.KE.flatten()[np.newaxis],self.device)))
      K=torch.sparse_coo_tensor(self.mesh.nodeIdx,sK,(self.mesh.ndof, self.mesh.ndof)).to_dense()
      return K

    def solve(K):
      # eliminate fixed dofs for solving sys of eqns
      u_free = torch.linalg.solve(K[self.free,:][:,self.free], \
              to_torch(np.array(self.f[self.free]),self.device)); #, sym_pos = True
      u = to_torch(np.zeros((self.mesh.ndof)),self.device)
      u[self.free]=u[self.free]+u_free.reshape(-1)
      # u = u.at[self.free].add(u_free.reshape(-1)) # homog bc wherev fixed
      # jax.debug.print("🤯 {x} 🤯", x=u)
      return u
    
    def computeCompliance(K, u):
      J = torch.dot(to_torch(self.f.reshape(-1).T,self.device), u)
      return J
    
    
    Y=getYoungsModulus(density, penal)
    K = assembleK(Y)
    u = solve(K)
    self.u2=u
    J = computeCompliance(K, u)
    return J


