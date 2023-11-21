# // Amira Abdel-Rahman 
# // (c) Massachusetts Institute of Technology 2023 

# based on https://github.com/UW-ERSL/JAXTOuNN
import numpy as np

class Material:
  def __init__(self, matProp):
    self.matProp = matProp
    E, nu = matProp['Emax'], matProp['nu'];
    self.C = E/(1-nu**2)* \
            np.array([[1, nu, 0],\
                      [nu, 1, 0],\
                      [0, 0, (1-nu)/2]]);
  #--------------------------#
  
  def computeSIMP_Interpolation(self, rho, penal):
    E = 0.001*self.matProp['Emax'] + \
            (0.999*self.matProp['Emax'])*\
            (rho+0.01)**penal
    return E
  #--------------------------#
  
  def computeRAMP_Interpolation(self, rho, penal):
    E = 0.001*self.matProp['Emax']  +\
        (0.999*self.matProp['Emax'])*\
            (rho/(1.+penal*(1.-rho)))
    return E
  #--------------------------#
  def getD0elemMatrix(self, mesh):
    dim=mesh.bcSettings['dofsPerNode']
    if(mesh.meshType == 'gridMesh'):
      E = 1
      nu = self.matProp['nu'];
      if dim==2:

        k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,\
                      -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
        D0 = E/(1-nu**2)*np.array\
            ([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
                # all the elems have same base stiffness

      elif dim==3:
        D0= self.lk_H8(E,nu)
      elif dim==1: #thermal
        D0= np.array(\
          [ 2./3., -1./6., -1./3., -1./6.,\
          -1./6.,  2./3., -1./6., -1./3.,\
          -1./3., -1./6.,  2./3., -1./6.,\
          -1./6., -1./3., -1./6.,  2./3.]);  
      return D0

    #--------------------------#

  
  def lk_H8(self,E,nu):
      
      A = np.array([[32,6,-8,6,-6,4,3,-6,-10,3,-3,-3,-4,-8],[-48,0,0,-24,24,0,0,0,12,-12,0,12,12,12]])
      b = np.array([[1],[nu]])
      k = 1/float(144)*np.dot(A.T,b).flatten()

      K1 = np.array([[k[0],k[1],k[1],k[2],k[4],k[4]],
      [k[1],k[0],k[1],k[3],k[5],k[6]],
      [k[1],k[1],k[0],k[3],k[6],k[5]],
      [k[2],k[3],k[3],k[0],k[7],k[7]],
      [k[4],k[5],k[6],k[7],k[0],k[1]],
      [k[4],k[6],k[5],k[7],k[1],k[0]]])

      K2 = np.array([[k[8],k[7],k[11],k[5],k[3],k[6]],
      [k[7],k[8],k[11],k[4],k[2],k[4]],
      [k[9],k[9],k[12],k[6],k[3],k[5]],
      [k[5],k[4],k[10],k[8],k[1],k[9]],
      [k[3],k[2],k[4],k[1],k[8],k[11]],
      [k[10],k[3],k[5],k[11],k[9],k[12]]])

      K3 = np.array([[k[5],k[6],k[3],k[8],k[11],k[7]],
      [k[6],k[5],k[3],k[9],k[12],k[9]],
      [k[4],k[4],k[2],k[7],k[11],k[8]],
      [k[8],k[9],k[1],k[5],k[10],k[4]],
      [k[11],k[12],k[9],k[10],k[5],k[3]],
      [k[1],k[11],k[8],k[3],k[4],k[2]]])

      K4 = np.array([[k[13],k[10],k[10],k[12],k[9],k[9]],
      [k[10],k[13],k[10],k[11],k[8],k[7]],
      [k[10],k[10],k[13],k[11],k[7],k[8]],
      [k[12],k[11],k[11],k[13],k[6],k[6]],
      [k[9],k[8],k[7],k[6],k[13],k[10]],
      [k[9],k[7],k[8],k[6],k[10],k[13]]])

      K5 = np.array([[k[0],k[1],k[7],k[2],k[4],k[3]],
      [k[1],k[0],k[7],k[3],k[5],k[10]],
      [k[7],k[7],k[0],k[4],k[10],k[5]],
      [k[2],k[3],k[4],k[0],k[7],k[1]],
      [k[4],k[5],k[10],k[7],k[0],k[7]],
      [k[3],k[10],k[5],k[1],k[7],k[0]]])

      K6 = np.array([[k[13],k[10],k[6],k[12],k[9],k[11]],
      [k[10],k[13],k[6],k[11],k[8],k[1]],
      [k[6],k[6],k[13],k[9],k[1],k[8]],
      [k[12],k[11],k[9],k[13],k[6],k[10]],
      [k[9],k[8],k[1],k[6],k[13],k[6]],
      [k[11],k[1],k[8],k[10],k[6],k[13]]])

      KE1=np.hstack((K1,K2,K3,K4))
      KE2=np.hstack((K2.T,K5,K6,K3.T))
      KE3=np.hstack((K3.T,K6,K5.T,K2.T))
      KE4=np.hstack((K4,K3,K2,K1.T))
      KE = E/float(((nu+1)*(1-2*nu)))*np.vstack((KE1,KE2,KE3,KE4))

      return(KE)