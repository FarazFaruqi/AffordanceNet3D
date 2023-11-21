# // Amira Abdel-Rahman 
# // (c) Massachusetts Institute of Technology 2023 

# based on https://github.com/UW-ERSL/JAXTOuNN
import numpy as np
import matplotlib.pyplot as plt


def getDOF3D(i,j,k,d,nelx,nely,nelz): #i,j,k from 0 to nelx,nely,nelz # d 0 x, 1 y , 2 z
  i=np.array(i).astype(int);j=np.array(j).astype(int);k=np.array(k).astype(int);
  dim=3
  el = j + (i * (nely+1)) + k * ((nelx+1)* (nely+1))
  return dim * el + d 


#  ~~~~~~~~~~~~ Examples ~~~~~~~~~~~~~#
   
def getExampleBC3D(example, nelx, nely, nelz):
    if(example == 1): # tip cantilever

        #  USER - DEFINED LOAD DOFs
        kl = np.arange(nelz + 1)
        loadnid = kl * (nelx + 1) * (nely + 1) + (nely + 1) * (nelx + 1) - 1  # Node IDs
        loaddof = 3 * loadnid + 1  # DOFs
        # USER - DEFINED SUPPORT FIXED DOFs
        [jf, kf] = np.meshgrid(np.arange(nely + 1), np.arange(nelz + 1))  # Coordinates
        fixednid = (kf) * (nely + 1) * (nelx + 1) + jf  # Node IDs
        fixed = np.array([3 * fixednid, 3 * fixednid + 1, 3 * fixednid + 2]).flatten()  # DOFs
        # BC's and support
        dofs = np.arange(3 * (nelx + 1) * (nely + 1) * (nelz + 1))
        free = np.setdiff1d(dofs, fixed)

        exampleName = 'TipCantilever'
        bcSettings = {'fixedNodes': fixed,\
                      'forceMagnitude': -0.1,\
                      'forceNodes': loaddof, \
                      'dofsPerNode':3};
        symMap = {'XAxis':{'isOn':False, 'midPt':0.5*nely},\
          'YAxis':{'isOn':False, 'midPt':0.5*nelx},\
            'ZAxis':{'isOn':False, 'midPt':0.5*nelz}}
    
    elif(example == 2): # table
        [il, kl] = np.meshgrid(np.arange(nelx + 1), np.arange(nelz + 1))  # Coordinates
        loaddof=getDOF3D(i=(il),j=(nely),k=(kl),d=1,nelx=nelx,nely=nely,nelz=nelz)

        iff=[0,0,nelx,nelx]
        kff=[0,nelz,0,nelz]
        fixed=np.array([getDOF3D(i=(iff),j=(0),k=(kff),d=0,nelx=nelx,nely=nely,nelz=nelz),\
                        getDOF3D(i=(iff),j=(0),k=(kff),d=1,nelx=nelx,nely=nely,nelz=nelz),\
                        getDOF3D(i=(iff),j=(0),k=(kff),d=2,nelx=nelx,nely=nely,nelz=nelz)]).flatten() 
        dofs = np.arange(3 * (nelx + 1) * (nely + 1) * (nelz + 1))
        free = np.setdiff1d(dofs, fixed)
       
       

        exampleName = 'table'
        bcSettings = {'fixedNodes': fixed,\
                      'forceMagnitude': -1.,\
                      'forceNodes': loaddof, \
                      'dofsPerNode':3};
        symMap = {'XAxis':{'isOn':False, 'midPt':0.5*nely},\
          'YAxis':{'isOn':False, 'midPt':0.5*nelx},\
            'ZAxis':{'isOn':False, 'midPt':0.5*nelz}}

    elif(example == 3): # table middle
        [il, kl] = np.meshgrid(np.arange(nelx + 1), np.arange(nelz + 1))  # Coordinates
        loaddof=getDOF3D(i=(il),j=(nely),k=(kl),d=1,nelx=nelx,nely=nely,nelz=nelz)

        iff=nelx/2
        kff=nelz/2
        fixed=np.array([getDOF3D(i=(iff),j=(0),k=(kff),d=0,nelx=nelx,nely=nely,nelz=nelz),\
                        getDOF3D(i=(iff),j=(0),k=(kff),d=1,nelx=nelx,nely=nely,nelz=nelz),\
                        getDOF3D(i=(iff),j=(0),k=(kff),d=2,nelx=nelx,nely=nely,nelz=nelz)]).flatten() 
        dofs = np.arange(3 * (nelx + 1) * (nely + 1) * (nelz + 1))
        free = np.setdiff1d(dofs, fixed)
       
       

        exampleName = 'tableMiddle'
        bcSettings = {'fixedNodes': fixed,\
                      'forceMagnitude': -1.,\
                      'forceNodes': loaddof, \
                      'dofsPerNode':3};
        symMap = {'XAxis':{'isOn':False, 'midPt':0.5*nely},\
          'YAxis':{'isOn':False, 'midPt':0.5*nelx},\
            'ZAxis':{'isOn':False, 'midPt':0.5*nelz}}
    elif(example == 4): # table new
        [il, kl] = np.meshgrid(np.arange(nelx + 1), np.arange(nelz + 1))  # Coordinates
        loaddof=getDOF3D(i=(il),j=(nely),k=(kl),d=1,nelx=nelx,nely=nely,nelz=nelz)

        # iff=[0,0,nelx,nelx]
        # kff=[0,nelz,0,nelz]
        # fixed=np.array([getDOF3D(i=(iff),j=(0),k=(kff),d=0,nelx=nelx,nely=nely,nelz=nelz),\
        #                 getDOF3D(i=(iff),j=(0),k=(kff),d=1,nelx=nelx,nely=nely,nelz=nelz),\
        #                 getDOF3D(i=(iff),j=(0),k=(kff),d=2,nelx=nelx,nely=nely,nelz=nelz)]).flatten() 
        
        [il, kl] = np.meshgrid(np.arange(nelx + 1), np.arange(nelz + 1))  # Coordinates
        fixed=getDOF3D(i=(il),j=(0),k=(kl),d=1,nelx=nelx,nely=nely,nelz=nelz)

        fixed=np.array([fixed,getDOF3D(i=(il),j=(1),k=(kl),d=1,nelx=nelx,nely=nely,nelz=nelz)]).flatten() 
        
        dofs = np.arange(3 * (nelx + 1) * (nely + 1) * (nelz + 1))
        free = np.setdiff1d(dofs, fixed)
       
       

        exampleName = 'table'
        bcSettings = {'fixedNodes': fixed,\
                      'forceMagnitude': -1.,\
                      'forceNodes': loaddof, \
                      'dofsPerNode':3};
        symMap = {'XAxis':{'isOn':False, 'midPt':0.5*nely},\
          'YAxis':{'isOn':False, 'midPt':0.5*nelx},\
            'ZAxis':{'isOn':False, 'midPt':0.5*nelz}}
    return exampleName, bcSettings, symMap

  