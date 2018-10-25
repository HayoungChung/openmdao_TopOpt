import numpy as np
import numpy.linalg as LA
import LinE as LinE
import Sensitivity_vec as Sens
#from slsm_Module import *
#from Opt_Module import *
# from LSM2d import pyLevelset
import matplotlib.pyplot as plt


class OptimRefact(object):
    def __init__(self, *args):
        super(OptimRefact, self).__init__()

        self.counter = 0

        self.cant = Cantilever(True)
        self.cant.set_fea()

        self.LSM = pyLevelset(self.cant.exy[0]+1,\
                                self.cant.exy[1]+1, 0.5)
        
        self.phi = np.zeros(self.cant.CMesh.nNodes)
        self.LSM.get_phi(self.phi)
        # print(np.sum(self.phi))

        self.areafraction = np.zeros(self.cant.CMesh.nElems)
        self.LSM.get_area_fractions(self.areafraction)

    def get_delphi(self,phi):
        self.phi = phi.reshape(self.cant.CMesh.nNodes)

        self.LSM.set_signedDistance(self.phi) # reinitialization...
        self.LSM.get_area_fractions(self.areafraction)
        u = self.cant.get_u(self.areafraction)

        nBpts = int(self.LSM.get_num_boundary_coords())
        BoundaryPoints = np.zeros((nBpts, 2))
        x = np.zeros(nBpts)
        y = np.zeros(nBpts)
        self.LSM.get_boundary_coords(x,y)
        # self.LSM.get_boundary_coords(BoundaryPoints[:,0],BoundaryPoints[:,1])
        # plt.plot(x,y,'o')
        BoundaryPoints[:,0] = x
        BoundaryPoints[:,1] = y
        # plt.show()
        
        bndSensitivities = self.cant.get_sens(BoundaryPoints)

        np.savetxt('txts/bndSensitivities_%i.txt' %self.counter, bndSensitivities, delimiter=' ')
        self.LSM.set_sensitivities(bndSensitivities);

        dphi_dt = np.zeros(self.cant.CMesh.nNodes)
    
        self.LSM.get_delphi(dphi_dt)
        dphi_dt *= -1.0;
	    # num_boundary_pts = self.cant.mesh.nNodes

        # np.savetxt('txts/signedDistance_%i.txt' %self.counter, self.phi, delimiter=' ')
        # np.savetxt('txts/velocities_%i.txt' %self.counter, dphi_dt, delimiter=' ')
        if 0:
            plt.clf()
            plt.plot(BoundaryPoints[:,0], BoundaryPoints[:,1], 'o')
            plt.savefig('plots/Bpts_%i.png' % self.counter)
    
            plt.clf()
            plt.scatter(BoundaryPoints[:,0],BoundaryPoints[:,1], 20, bndSensitivities, marker = 'o')
            plt.savefig('plots/Bsens_%i.png' % self.counter)
    
            plt.clf()
            plt.scatter(self.cant.CMesh.Nodes[:,0], self.cant.CMesh.Nodes[:,1], 20, dphi_dt, marker = 'o')
            plt.savefig('plots/dphi_dt_%i.png' % self.counter)
    
            plt.clf()
            plt.scatter(self.cant.CMesh.Nodes[:,0], self.cant.CMesh.Nodes[:,1], 20, self.phi, marker = 'o')
            plt.savefig('plots/phi_%i.png' % self.counter)
            self.counter += 1
            print("counter %i" % self.counter)
            
        # plt.show()
        self.LSM.reinitialize()
        return dphi_dt
	

class Cantilever(object):
    def __init__(self, isHoles = False,lxy = [160,80], exy = [160,80], *args):
        super(Cantilever, self).__init__()
        self.lxy = lxy# = [160, 80]
        self.exy = exy#= [160, 80]
        # nELEM = 160*80
        # levelset option
        # AllowedAreaFraction = 0.01
        # moveLimit = 0.9
        # VoidMaterial = 1e-6

        self.CMesh = LinE.FEAMeshQ4(lxy, exy)
        self.CMesh.nNodes = (exy[0]+1)*(exy[1]+1)
        self.CMesh.nElems = (exy[0])*(exy[1])
        # print("lxy = " + str(lxy))

    def set_fea(self, *args):
        lxy = self.lxy
        # print ("setup fea")

        E = 1.0
        v = 0.3
        thickness = 1.0
        # print("(E,v,h) = " + str(E) + "," + str(v) + "," + str(thickness))

        Cijkl = LinE.LinearElasticMaterial.get_Cijkl_E_v(E, v)
        xtip1 = self.CMesh.get_NodeID([lxy[0], int(lxy[1]/2)], 1e-3, 1e-3)
        xtip2 = self.CMesh.get_NodeID([lxy[0], int(lxy[1]/2)-1], 1e-3, 1e-3)
        xtip3 = self.CMesh.get_NodeID([lxy[0], int(lxy[1]/2)+1], 1e-3, 1e-3)

        self.__BC_force1 = self.CMesh.get_dof('y', xtip1)
        self.__BC_force2 = self.CMesh.get_dof('y', xtip2)
        self.__BC_force3 = self.CMesh.get_dof('y', xtip3)

        self.CLinElasticity = LinE.LinearElasticity(self.CMesh, Cijkl)
        self.CSensitivities = Sens.ElasticitySensitivities(self.CLinElasticity)

    def get_u(self, areafraction):

        nELEM = self.CMesh.nELEM
        self.CMesh.AreaFraction = np.zeros(nELEM)
        for ii in range(0,nELEM):
            if areafraction[ii] < 1e-6: #AllowedAreaFraction
                self.CMesh.AreaFraction[ii] = 1e-6 #VoidMaterial
            else:
                self.CMesh.AreaFraction[ii] = areafraction[ii]

        self.CLinElasticity.Assembly() 
        # print ("... done")
        Xlo_id = self.CMesh.get_NodeID([0,0],1e-3,np.inf)
        BC_fixed = self.CMesh.get_dof('xy',Xlo_id)

        self.CLinElasticity.Apply_BC(BC_fixed)

        self.CLinElasticity.set_F()
        self.CLinElasticity.set_F(self.__BC_force1,-1)

        #self.CLinElasticity.set_F(self.__BC_force2,-2.5)
        #self.CLinElasticity.set_F(self.__BC_force3,-2.5)

        # print ("computing displacement")
        u = self.CLinElasticity.solve()

        # NodesF = self.CMesh.Nodes + u.reshape(2,self.CMesh.nNODE,order='F').transpose()
        # idE = self.CMesh.Elements[:,[0,1,2,3,0]].flatten(order='C').astype(int)
        # xorder = self.CMesh.Nodes[idE,0].reshape(int(len(idE)/(self.CMesh._npe+1)),self.CMesh._npe+1)
        # yorder = self.CMesh.Nodes[idE,1].reshape(int(len(idE)/(self.CMesh._npe+1)),self.CMesh._npe+1)
        # plt.plot(xorder.transpose(),yorder.transpose())
        # plt.savefig("DEFORM.png")

        
        return u,
        
    # def set_sens(self, *args):
    #     self.CSensitivities = Sens.ElasticitySensitivities(self.CLinElasticity)
    #         
    def get_sens(self, BoundaryPoints, *args):
        # sensitivity option
        Radius = 2
        Weights = 1
        AllowedAreaFraction = 0.01
        WeightFlag = 5
        BoundarySensitivities = self.CSensitivities.Compliance(BoundaryPoints, Weights, Radius, WeightFlag, AllowedAreaFraction)
        return BoundarySensitivities
        
    def get_IntegPoints(self):
        return self.CSensitivities.IntegrationPoints

    def get_IntegPoints_xy(self): ## TEMP
        na = np.zeros((160*80*4,2))
        for ee in range(0,160*80):
            for gg in range(0,4):
                na[ee*4+gg,0] = self.CSensitivities.IntegrationPoints[ee,gg,0]
                na[ee*4+gg,1] = self.CSensitivities.IntegrationPoints[ee,gg,1]
        return na
