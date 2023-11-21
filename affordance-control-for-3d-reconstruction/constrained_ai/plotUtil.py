# // Amira Abdel-Rahman 
# // (c) Massachusetts Institute of Technology 2023 


import matplotlib.pyplot as plt
import numpy as np


import trimesh
from skimage import measure
import plotly.graph_objects as go


def plotFieldOnMesh( field, titleStr="",iso=0.1,alphaBackground=0,name=0):
    nelz,nelx,nely=field.shape

    lighting_effects = dict(ambient=0.4, diffuse=0.9, roughness = 0.9, specular=1.0, fresnel=0.2)
    vol=np.zeros((2+nelz,2+nelx,2+nely))
    vol[1:-1,1:-1,1:-1]=field.reshape((nelz,nelx,nely))
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

    # if not os.path.exists("images"):
    #     os.mkdir("images")
    # fig.write_image("images/"+str(name)+".png")


"""# Display Displacements"""

#3rd party widgets need to be enabled during a session

#widget modules
from ipywidgets import interact, interactive


def plotDisplacement(mesh,density,u):
    nelx=mesh.nelx
    nely=mesh.nely
    bc=mesh.bc
    edofMat=mesh.edofMat
    bc["uTarget"]=[]
    xyElems = jnp.array(mesh.generatePoints())
    umat=np.array(u[edofMat])
    meshplot=np.zeros((xyElems.shape[0],4,2))
    meshplot[:,0,0]=xyElems[:,0]-0.5
    meshplot[:,0,1]=xyElems[:,1]+0.5

    meshplot[:,1,0]=xyElems[:,0]+0.5
    meshplot[:,1,1]=xyElems[:,1]+0.5

    meshplot[:,2,0]=xyElems[:,0]+0.5
    meshplot[:,2,1]=xyElems[:,1]-0.5

    meshplot[:,3,0]=xyElems[:,0]-0.5
    meshplot[:,3,1]=xyElems[:,1]-0.5


    def displayDisplacement(factor):
    

        fig = plt.figure(3,figsize=(8, 8))
        ax = fig.add_subplot(111)
        
        for i in range(nelx*nely):
            c='black'
            if np.sum(np.isin(bc["fixed"], edofMat[i]))>0:
                c="red"
            elif np.sum(np.isin(np.nonzero(bc["force"])[0], edofMat[i]))>0:
                c="blue"
                ii=np.where(edofMat[i]==np.nonzero(bc["force"])[0])[0]
                xx=meshplot[i,ii//2,0]+umat[i,ii//2*2]*factor;yy=meshplot[i,ii//2,1]+umat[i,ii//2*2+1]*factor;
                if ii%2==0:
                    xytext=(xx-bc["force"][np.nonzero(bc["force"])[0]], yy)
                else:
                    xytext=(xx, yy-bc["force"][np.nonzero(bc["force"])[0]])
                ax.annotate("", xy=(xx, yy), xytext=xytext, arrowprops=dict(arrowstyle='-|>, head_width=1, head_length=1'),)
            elif np.sum(np.isin(bc["uTarget"], edofMat[i]))>0:
                c="green"

            ax.fill(meshplot[i,:,0].T+umat[i,[0,2,4,6]].T*factor,meshplot[i,:,1].T+umat[i,[1,3,5,7]].T*factor,facecolor=c,alpha=density[i]*0.8)
            if np.sum(np.isin(bc["uTarget"], edofMat[i]))>0:
                iii=np.where(np.isin(bc["uTarget"], edofMat[i])==True)[0]
                ii=np.where(edofMat[i]==bc["uTarget"][iii[0]])[0]
                ax.scatter(meshplot[i,ii//2,0]+bc["uTargetv"][iii[0]],meshplot[i,ii//2,1]+bc["uTargetv"][iii[1]] ,200,c="green", marker='x')

        ax.set_aspect('equal')
        plt.xlim([0, nelx*1.2])
        plt.ylim([0, nely*1.2])
        fig.show()
    interactive_plot = interactive(displayDisplacement,factor=(0.00,1,0.001))
    # output = interactive_plot.children[-1]
    return interactive_plot


