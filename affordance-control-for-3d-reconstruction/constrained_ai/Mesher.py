# // Amira Abdel-Rahman 
# // (c) Massachusetts Institute of Technology 2023 

# based on https://github.com/UW-ERSL/JAXTOuNN


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go

from skimage import measure
# from skimage.draw import ellipsoid
from IPython.display import clear_output
import trimesh
import kaleido

import numpy.matlib as npm


def iKjK3(nelx,nely,nelz):
    edofMat = np.zeros((nelx * nely * nelz, 24), dtype=int)
    for elz in range(nelz):
        for elx in range(nelx):
            for ely in range(nely):
                el = ely + (elx * nely) + elz * (nelx * nely)
                n1 = elz * (nelx + 1) * (nely + 1) + (nely + 1) * elx + ely
                n2 = elz * (nelx + 1) * (nely + 1) + (nely + 1) * (elx + 1) + ely
                n3 = (elz + 1) * (nelx + 1) * (nely + 1) + (nely + 1) * elx + ely
                n4 = (elz + 1) * (nelx + 1) * (nely + 1) + (nely + 1) * (elx + 1) + ely
                edofMat[el, :] = np.array(
                    [3 * n1 + 3, 3 * n1 + 4, 3 * n1 + 5, 3 * n2 + 3, 3 * n2 + 4, 3 * n2 + 5, \
                      3 * n2, 3 * n2 + 1, 3 * n2 + 2, 3 * n1, 3 * n1 + 1, 3 * n1 + 2, \
                      3 * n3 + 3, 3 * n3 + 4, 3 * n3 + 5, 3 * n4 + 3, 3 * n4 + 4, 3 * n4 + 5, \
                      3 * n4, 3 * n4 + 1, 3 * n4 + 2, 3 * n3, 3 * n3 + 1, 3 * n3 + 2])
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((24, 1))).flatten().astype(int)
    jK = np.kron(edofMat, np.ones((1, 24))).flatten().astype(int)
    return iK,jK,edofMat


class RectangularGridMesher3D:
  #--------------------------#
  def __init__(self, ndim, nelx, nely, nelz, elemSize, bcSettings):
    self.meshType = 'gridMesh'
    self.ndim = ndim;
    self.nelx = nelx;
    self.nely = nely;
    self.nelz = nelz;
    self.elemSize = elemSize;
    self.bcSettings = bcSettings;
    self.numElems = self.nelx*self.nely*self.nelz;
    self.elemArea = self.elemSize[0]*self.elemSize[1]*self.elemSize[2]*\
                    np.ones((self.numElems)) # all same areas for grid
    self.totalMeshArea = np.sum(self.elemArea);
    self.numNodes = (self.nelx+1)*(self.nely+1)*(self.nelz+1);
    self.nodesPerElem = 8; # grid cube mesh
    self.ndof = self.bcSettings['dofsPerNode']*self.numNodes;
    self.edofMat, self.nodeIdx, self.elemNodes, self.nodeXY, self.bb = \
                                    self.getMeshStructure();
    self.elemCenters = self.generatePoints();
    self.processBoundaryCondition();
    # self.BMatrix = self.getBMatrix(0., 0.)
    # self.fig, self.ax = plt.subplots()
  #--------------------------#
  

    iK,jK,edofMat=iKjK3(nelx,nely,nelz)
  #--------------------------#
  def getMeshStructure(self):
    # returns edofMat: array of size (numElemsX8) with
    # the global dof of each elem
    # idx: A tuple informing the position for assembly of computed entries
    n = self.bcSettings['dofsPerNode']*self.nodesPerElem;
    iK,jK,edofMat=iKjK3(self.nelx,self.nely,self.nelz)
    nodeIdx = (iK,jK)
    elemNodes = np.zeros((self.numElems, self.nodesPerElem));
    for elz in range(self.nelz):
      for elx in range(self.nelx):
        for ely in range(self.nely):
          el = ely + (elx * self.nely) + elz * (self.nelx * self.nely)
          n1 = elz * (self.nelx + 1) * (self.nely + 1) + (self.nely + 1) * elx + ely
          n2 = elz * (self.nelx + 1) * (self.nely + 1) + (self.nely + 1) * (elx + 1) + ely
          n3 = (elz + 1) * (elx + 1) * (self.nely + 1) + (self.nely + 1) * elx + ely
          n4 = (elz + 1) * (elx + 1) * (self.nely + 1) + (self.nely + 1) * (elx + 1) + ely

          elemNodes[el,:] = np.array([n1+1, n2+1, n3+1, n4+1, n4, n3, n2, n1]) #TODO check as not sure if correct 3d


    bb = {}
    bb['xmin'],bb['xmax'],bb['ymin'],bb['ymax'],bb['zmin'],bb['zmax'] = \
        0., self.nelx*self.elemSize[0],\
        0., self.nely*self.elemSize[1],\
        0., self.nelz*self.elemSize[2]
        
    nodeXYZ = np.zeros((self.numNodes, 3))
    ctr = 0;
    for k in range(self.nelz+1):
      for i in range(self.nelx+1):
        for j in range(self.nely+1):
          nodeXYZ[ctr,0] = self.elemSize[0]*i;
          nodeXYZ[ctr,1] = self.elemSize[1]*j;
          nodeXYZ[ctr,2] = self.elemSize[2]*k;
          ctr += 1;
            
    return edofMat, nodeIdx, elemNodes, nodeXYZ, bb
  #--------------------------#

  def generatePoints(self, res=1):
    # args: Mesh is dictionary containing nelx, nely, elemSize...
    # res is the number of points per elem
    # returns an array of size (numpts X 2)
    xyz = np.zeros((res**3*self.numElems, 3))
    ctr = 0
    for k in range(res*self.nelz):
      for i in range(res*self.nelx):
        for j in range(res*self.nely):
          xyz[ctr, 0] = (i + 0.5)/(res*self.elemSize[0])
          xyz[ctr, 1] = (j + 0.5)/(res*self.elemSize[1])
          xyz[ctr, 2] = (k + 0.5)/(res*self.elemSize[2])
          ctr += 1
    return xyz
  #--------------------------#
  def processBoundaryCondition(self):
    force = np.zeros((self.ndof,1))
    dofs=np.arange(self.ndof)
    fixed = dofs[self.bcSettings['fixedNodes']]
    free = np.setdiff1d(np.arange(self.ndof), fixed)
    force[self.bcSettings['forceNodes']] = self.bcSettings['forceMagnitude']
    self.bc = {'force':force, 'fixed':fixed,'free':free}
  #--------------------------#
  def exportMesh(self, field, name="mesh.stl",iso=0.5):
    lighting_effects = dict(ambient=0.4, diffuse=0.5, roughness = 0.9, specular=0.9, fresnel=0.2)
    vol=np.zeros((2+self.nelz,2+self.nelx,2+self.nely))
    vol[1:-1,1:-1,1:-1]=field.reshape((self.nelz,self.nelx,self.nely))
    verts, faces, normals, values = measure.marching_cubes(vol, iso)
    mesh1 = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    trimesh.repair.fix_inversion(mesh1)
    trimesh.repair.fill_holes(mesh1)
    trimesh.repair.fix_normals(mesh1)
    mesh1.export(name)

  def plotFieldOnMesh(self, field, titleStr="",iso=0.1,alphaBackground=0,name=0):
    # plt.ion(); plt.clf()
    # plt.imshow(-np.flipud(field.reshape((self.nelx,self.nely)).T), \
    #            cmap='gray', interpolation='none')
    # plt.axis('Equal')
    # plt.grid(False)
    # plt.title(titleStr)
    # plt.pause(0.01)
    # self.fig.canvas.draw()

    # X, Y, Z = np.mgrid[0:nelz , 0:nelx , 0:nely ]
    # vol =  xPhys.reshape((nelz,nelx,nely))


    # fig = go.Figure(data=go.Isosurface(
    #     x=X.flatten(),
    #     y=Y.flatten(),
    #     z=Z.flatten(),
    #     value=vol.flatten(),
    #     surface=dict(count=2, fill=1.0),
    #     isomin=0.7,
    #     isomax=0.95,
    #     caps=dict(x_show=True, y_show=True)
    #     ))

    clear_output()
    lighting_effects = dict(ambient=0.4, diffuse=0.9, roughness = 0.9, specular=1.0, fresnel=0.2)
    vol=np.zeros((2+self.nelz,2+self.nelx,2+self.nely))
    vol[1:-1,1:-1,1:-1]=field.reshape((self.nelz,self.nelx,self.nely))
    verts, faces, normals, values = measure.marching_cubes(vol, iso)


    fig = go.Figure(go.Mesh3d(
            x=verts[:,0], y=verts[:,1], z=verts[:,2], 
            i=faces[:,0], j=faces[:,1], k=faces[:,2],
            lighting=lighting_effects,
            
            color="teal"))

    camera = dict(
              up=dict(x=0, y=0, z=1),
              center=dict(x=0, y=0, z=0),
              eye=dict(x=0.1, y=2.5, z=0.1))
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=1.25, z=0.1))

    fig.update_layout(
        title=titleStr,
        scene_camera = camera,
        paper_bgcolor='rgba(255,255,255,'+str(alphaBackground)+')',
        font_color="black",
        scene = dict(
                      xaxis = dict(
                            backgroundcolor="rgba(0, 0, 0,0)",
                            gridcolor="rgba(0, 0, 0,0)",
                            showbackground=False,
                            zerolinecolor="black",),
                      yaxis = dict(
                          backgroundcolor="rgba(0, 0, 0,0)",
                          gridcolor="black",
                          showbackground=False,
                          zerolinecolor="black"),
                      zaxis = dict(
                          backgroundcolor="rgba(0, 0, 0,0)",
                          gridcolor="black",
                          showbackground=False,
                          zerolinecolor="black",),),)

    fig['layout']['scene']['aspectmode'] = "data"
    # fig.update_xaxes(range=[-1, self.nelz+1])
    # fig.update_yaxes(range=[-1, self.nelx+1])
    # fig.update_zaxes(range=[-1, nely+1])
    fig.show()
    import os

    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/"+str(name)+".png")
