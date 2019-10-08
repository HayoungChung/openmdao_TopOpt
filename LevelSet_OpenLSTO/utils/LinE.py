# python version of LinearElasticity
# check through test_LinE.py, which reproduces FEA results (cantilever)
# TODO: lil matrix should be used for efficiency
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

class FEAMeshQ4(object):
    '''it works'''
    _dpn = 2
    _npe = 4
    _dpe = _dpn*_npe
    
    def __init__(self, *args):
        super(FEAMeshQ4,self).__init__()
        lxy = args[0]
        exy = args[1]
        
        (self.Nodes, self.Elements) = self.get_Mesh(lxy, exy)
        self.nELEM = self.Elements.shape[0]
        self.nNODE = self.Nodes.shape[0]
        
        self.get_Area()
        self.AreaFraction = np.copy(self.Area)
                                                
    def get_Area(self):
        ''' http://www.mathopenref.com/coordpolygonarea2.html'''
        self.Area = np.zeros(self.nELEM)
        for ii in range(0,self.nELEM):
            elem_id = self.Elements[ii,:].astype(int)
            x_list = self.Nodes[elem_id[[0,1,2,3,0]],0]
            y_list = self.Nodes[elem_id[[0,1,2,3,0]],1]
            QuadArea_ = 0
            for qq in range(0,4):
                QuadArea_ += 0.5*((x_list[qq]+x_list[qq+1])\
                                        *(y_list[qq]-y_list[qq+1]))
            self.Area[ii] = np.abs(QuadArea_)
            
        
    def get_Mesh(self,lxy,exy):
        # following x, y order from m2do: x first 
        (lx,ly) = lxy
        (ex,ey) = exy
        (nx,ny) = (ex+1, ey+1)        
        
        self.Centeroids = np.zeros((ex*ey,2))        
        Nodes = np.zeros([nx*ny, self._dpn])
        
        [x,y] = np.meshgrid(np.linspace(0,lx,nx),np.linspace(0,ly,ny),indexing='ij')
        
        Nodes[:,0] = x.flatten(order='F')
        Nodes[:,1] = y.flatten(order='F')
        
        Elements = np.zeros((ex*ey,self._npe))
        eid = 0
        
        for yy in range(0,ey):
            for xx in range(0,ex):
                Elements[eid,0] = nx*(yy)   + (xx)
                Elements[eid,1] = nx*(yy)   + (xx+1)
                Elements[eid,2] = nx*(yy+1) + (xx+1)
                Elements[eid,3] = nx*(yy+1) + (xx)
                eid += 1
                
        for ee in range(0,ex*ey):
            eid = Elements[ee,:].astype(int)
            X = Nodes[eid,:]
            self.Centeroids[ee,:] = sum(X)*0.25
                    
        return (Nodes, Elements)
        
    @staticmethod
    def get_N(r,s,order=2):
        N1 = 1/4.*(1+r)*(1+s)
        N2 = 1/4.*(1-r)*(1+s)
        N3 = 1/4.*(1-r)*(1-s)
        N4 = 1/4.*(1+r)*(1-s)
        
        N_o = np.array([N1,N2,N3,N4])
        return N_o 

    @staticmethod
    def get_N_rs(r,s,order=2):
        Ni_r = np.array([s+1,-s-1,s-1,1-s])/4.
        Ni_s = np.array([r+1,1-r,r-1,-r-1])/4.
        
        Ni_rs_o = np.vstack((Ni_r,Ni_s))
        return Ni_rs_o
    
    @staticmethod
    def get_gpts(IntegrationPoints):
        if IntegrationPoints == 1:
            ri = 0.
            si = 0.
            wi = 4.
        elif IntegrationPoints == 2:
            ri = np.array([-1., 1., 1., -1.])/np.sqrt(3)
            si = np.array([-1., -1., 1., 1.])/np.sqrt(3)
            wi = np.array([1., 1., 1, 1. ])
        elif IntegrationPoints == 3:
            ri = np.array([-1., 0., 1., -1., 0., 1., -1., 0., 1.])*np.sqrt(3/5)
            si = np.array([-1., -1., -1.,  0., 0., 0.,  1., 1., 1.])*np.sqrt(3/5)
            wi = np.array([ 1., 0.,   1.,  0., 0., 0.,  1., 0., 1.])*25./81. +  \
                 np.array([ 0., 1.,   0.,  1., 0., 1.,  0., 1., 0.])*40./81. + \
                 np.array([ 0., 0.,   0.,  0., 1., 0.,  0., 0., 0.])*64./81.
        else:
            print('get_gpts must be extended')
        return ri, si, wi
        
    def get_dof(self,dirStr,ID_nodes):
        ID_out = []
        for idx in ID_nodes:
            if "x" in str.lower(dirStr):
                ID_out.append(idx*self._dpn)
            if "y" in str.lower(dirStr):
                ID_out.append(idx*self._dpn + 1)
            ID_out.sort()  
        return ID_out
                
    def get_NodeID(self,position,Xtol,Ytol):
        id_x = np.nonzero(np.abs(self.Nodes[:,0]-position[0]) < Xtol)[0]
        id_y = np.nonzero(np.abs(self.Nodes[:,1]-position[1]) < Ytol)[0]       
        id_o = np.intersect1d(id_x,id_y)
        return np.sort(np.unique(id_o))
    
    def plot_mesh(self):
        idE = self.Elements[:,[0,1,2,3,0]].flatten(order='C').astype(int)
        xorder = self.Nodes[idE,0].reshape(int(len(idE)/(self._npe+1)),self._npe+1)
        yorder = self.Nodes[idE,1].reshape(int(len(idE)/(self._npe+1)),self._npe+1)
        self.p = plt.plot(xorder.transpose(),yorder.transpose(),'b-')
        

class LinearElasticity(object):
    def __init__(self,CMesh,Cijkl,*args):
        super(LinearElasticity,self).__init__()
        self.CMesh = CMesh
        self.Cijkl = Cijkl
        self.isBC = False
        nDOF = self.CMesh.nNODE*self.CMesh._dpn
        self.NumGpts = 2

        self.F  = np.zeros(nDOF)

    def Assembly(self):
        nDOF = self.CMesh.nNODE*self.CMesh._dpn
        idi = np.zeros(self.CMesh._dpe**2*self.CMesh.nELEM)
        idj = np.zeros(self.CMesh._dpe**2*self.CMesh.nELEM)
        datij = np.zeros(self.CMesh._dpe**2*self.CMesh.nELEM)
        
        (ri,si,wi) = self.CMesh.get_gpts(self.NumGpts)
        
        for ee in range(0,self.CMesh.nELEM):
            elem_id = self.CMesh.Elements[ee,:].astype(int)
            elem_dof = np.vstack((np.array(elem_id*2),np.array(elem_id*2+1)))\
                                .flatten(order='F')
            X = self.CMesh.Nodes[elem_id,:]
            
            LK = np.zeros((self.CMesh._dpe,self.CMesh._dpe))

            for gg in range(0,len(wi)):
                (r,s,w) = [ri[gg],si[gg],wi[gg]]
                N = self.CMesh.get_N(r,s)
                N_rs = self.CMesh.get_N_rs(r,s)
                
                matJ = N_rs.dot(X)
                
                N_XY = np.linalg.inv(matJ).dot(N_rs)
                N_X = N_XY[0,:]
                N_Y = N_XY[1,:]
                
                matB = np.zeros((3,self.CMesh._dpe))
                matB[0,0::self.CMesh._dpn] = N_X
                matB[1,1::self.CMesh._dpn] = N_Y
                matB[2,0::self.CMesh._dpn] = N_Y          
                matB[2,1::self.CMesh._dpn] = N_X
                    
                Jw = np.linalg.det(matJ)*w
                LK += matB.transpose().dot(self.Cijkl).dot(matB)*Jw

            LK *= self.CMesh.AreaFraction[ee]
            
            (idx_,idy_) = np.meshgrid(elem_dof,elem_dof)
            sparseid = range(ee*(self.CMesh._dpe**2),(ee+1)*(self.CMesh._dpe**2))
            idi[sparseid] = idx_.flatten(order = 'F')
            idj[sparseid] = idy_.flatten(order = 'F')
            datij[sparseid] = LK.flatten(order = 'F')
        
        self.sK = sp.sparse.csr_matrix((datij,(idi,idj)),shape=(nDOF,nDOF))
                
    def Apply_BC(self,DoF_id):
        self.sK[DoF_id,:] = 0
        self.sK[:,DoF_id] = 0
        self.sK[np.ix_(DoF_id,DoF_id)] = np.eye(len(DoF_id))
        self.isBC = True
        pass
    
    def set_F(self,*args):
        if len(args)==2:
            DoF_id = args[0]
            value = args[1]
            for id_dof in DoF_id:
                self.F[id_dof] += value
        else:
            self.F = np.zeros(self.CMesh.nNODE*self.CMesh._dpn)
    def solve(self):
        if self.isBC == False:
            print ("BC should be applied first\n")
            pass
        else:
            self.Field = spsolve(self.sK,self.F)
        return self.Field

    def get_compliance(self,Field, Weights=1):
            self.WeightedCompliance = Weights * (self.Field.dot(self.F))        
            return self.WeightedCompliance

class LinearElasticMaterial(object):
    def __init__(self):
        super(LinearElasticMaterial,self).__init__()        
    
    @staticmethod
    def get_Cijkl_E_v(c_E,c_v):
        Cijkl = np.array([[1,c_v,0],[c_v,1,0],[0,0,(1-c_v)/2]])*c_E/(1-c_v**2)
        return Cijkl
    
    @staticmethod
    def get_Cijl_lambda_mu(c_lambda,c_mu):
        Cijkl = np.diag([2,2,1])*c_mu + \
                np.array([[1,1,0],[1,1,0],[0,0,0]])*c_lambda
        return Cijkl
