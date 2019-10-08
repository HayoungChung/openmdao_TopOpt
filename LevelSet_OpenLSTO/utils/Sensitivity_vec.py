# Pythonized ver . of sensitivity analysis
# self.CLinElasticity.NumGpts**2 -> 2 is dimension
import numpy as np

class SensitivityAnalysis(object):
    def __init__(self,CLinElasticity,Tolerance):
        self.CLinElasticity = CLinElasticity
        
        # global coord of integration point 
        self.IntegrationPoints = np.zeros([self.CLinElasticity.CMesh.nELEM,self.CLinElasticity.NumGpts**2,2])
        
        # B matrices for each gpts
        self.B_gpts = np.zeros([self.CLinElasticity.CMesh.nELEM,self.CLinElasticity.NumGpts**2,3,self.CLinElasticity.CMesh._dpe]) 
        
        self.get_IntegrationPointData(Tolerance)
    
        #1. define a radius 5/4: # conservative measure
        
    def get_IntegrationPointData(self,Tolerance):
        (ri,si,wi) = self.CLinElasticity.CMesh.get_gpts(self.CLinElasticity.NumGpts)
        
        for ee in range(0,self.CLinElasticity.CMesh.nELEM):
#            if self.CLinElasticity.CMesh.AreaFraction[ee] <= Tolerance:
#                continue
        
            elem_id = self.CLinElasticity.CMesh.Elements[ee,:].astype(int)
#            elem_dof = np.vstack((np.array(elem_id*2),np.array(elem_id*2+1)))\
#                            .flatten(order='F')
            X = self.CLinElasticity.CMesh.Nodes[elem_id,:]
#            u = Field[elem_dof]          
            
            for gg in range(0,len(wi)):            
                (r,s) = [ri[gg],si[gg]]
                N = self.CLinElasticity.CMesh.get_N(r,s)
                self.IntegrationPoints[ee,gg,:] = N.dot(X)

                N_rs = self.CLinElasticity.CMesh.get_N_rs(r,s)
                
                matJ = N_rs.dot(X)
                
                N_XY = np.linalg.inv(matJ).dot(N_rs)
                N_X = N_XY[0,:]
                N_Y = N_XY[1,:]
                
                matB = np.zeros((3,self.CLinElasticity.CMesh._dpe))
                matB[0,0::self.CLinElasticity.CMesh._dpn] = N_X
                matB[1,1::self.CLinElasticity.CMesh._dpn] = N_Y
                matB[2,0::self.CLinElasticity.CMesh._dpn] = N_Y          
                matB[2,1::self.CLinElasticity.CMesh._dpn] = N_X
                
                self.B_gpts[ee,gg,:,:] = matB

        
    def IntegrationPointFieldGradients(self,Tolerance,*args):
        # xx,yy,xy strains: compute B*u for each integration point
        self.FieldGradient = np.zeros([self.CLinElasticity.CMesh.nELEM,self.CLinElasticity.NumGpts**2,3]) 
        
        if len(args) == 1:
            Field = args[0]
        else:
            Field = self.CLinElasticity.Field
        
        for ee in range(0,self.CLinElasticity.CMesh.nELEM):
            if self.CLinElasticity.CMesh.AreaFraction[ee] <= Tolerance:
                continue
            elem_id = self.CLinElasticity.CMesh.Elements[ee,:].astype(int)
            elem_dof = np.vstack((np.array(elem_id*2),np.array(elem_id*2+1)))\
                                .flatten(order='F')
#            X = self.CLinElasticity.CMesh.Nodes[elem_id,:]
            u = Field[elem_dof]        
            for gg in range(0,self.CLinElasticity.NumGpts**2):            
                self.FieldGradient[ee,gg,:] = self.B_gpts[ee,gg,:,:].dot(u)
                
    def ComputeBoundaryPointSensitivities(self,BoundaryPoints,Sensitivities \
                            ,Radius = 2.0, Weightflag = 4, Tolerance = 0.001):
        # computes BP sensitivities for giben Points coordinates (Bound-Points)        
        # tricky one: least-square metod
        p = int( ((3.14*Radius**2) /self.CLinElasticity.CMesh.Area[0]) * self.CLinElasticity.NumGpts**2 * (5./4.) )
        Distances = np.zeros(p)
        Indices = np.zeros((p,2)).astype(int)
        PointSensitivities = 0.
        
        # these should be calculated        
        # BoundaySensitivities 
        # IntegrationPointSenstitivities
                
        # 2. compute the # of integration points 
        # first remove elements when their centeroid is farther than 1.5Rad
        # build elementIndices, Distances, # indices
        CntPoints = int(0)

#        for ee in range(0,self.CLinElasticity.CMesh.nELEM):
            # elem_id = self.CLinElasticity.CMesh.Elements[ee,:].astype(int)
            # X = self.CLinElasticity.CMesh.Nodes[elem_id,:]
            # sum(X)/self.CLinElasticity.CMesh._npe # TOO SLOW
        el_cood = self.CLinElasticity.CMesh.Centeroids
        el_dist = np.sqrt(np.sum( ( np.tile(BoundaryPoints,(self.CLinElasticity.CMesh.nELEM,1)) - el_cood )**2, axis=1)) #TOFIX            
        id_e_within = np.nonzero(el_dist < 1.5*Radius)[0]
        ee_tmp_ = 0
        CntPoints = 0
        for ee in id_e_within:
            gg_dist = np.sqrt(np.sum((np.tile(BoundaryPoints,(self.IntegrationPoints.shape[1],1))-self.IntegrationPoints[ee,:,:])**2,axis = 1))
            id_g_within = np.nonzero(gg_dist < Radius)[0]
            CntPoints += len(id_g_within)
            Distances[ee_tmp_:CntPoints] = gg_dist[id_g_within]
            Indices[ee_tmp_:CntPoints,0] = ee
            Indices[ee_tmp_:CntPoints,1] = id_g_within
            ee_tmp_ += len(id_g_within)
            
        if CntPoints < 10:
            print("a very small island is found\n")
            PointSensitivities = 0
        
        A = np.zeros((CntPoints,6))
        b_sens = np.zeros(CntPoints)
        Bmax = 1e20
        Bmin = -1e20

        if Weightflag == 1:
            temp = 1 # least squares
        elif Weightflag == 2:
            temp = 1/Distances[0:CntPoints]
        elif Weightflag == 3:
            temp = self.CLinElasticity.CMesh.AreaFraction[Indices[0:CntPoints,0]]
        elif Weightflag == 4:
            temp = self.CLinElasticity.CMesh.AreaFraction[Indices[0:CntPoints,0]]/Distances[0:CntPoints]
        elif Weightflag == 5:
            temp = np.sqrt(self.CLinElasticity.CMesh.AreaFraction[Indices[0:CntPoints,0]]/Distances[0:CntPoints])
        else:
            temp = 1
            print("Weight Flag should lie in [1, 5]. Using Least Squares.\n") 

        for nn in range(0,CntPoints):
            RelativeCoordinate = self.IntegrationPoints[Indices[nn,0],Indices[nn,1]] - BoundaryPoints
            A[nn,0] = temp[nn]
            A[nn,1] = RelativeCoordinate[0]*temp[nn]
            A[nn,2] = RelativeCoordinate[1]*temp[nn]                 
            A[nn,3] = (RelativeCoordinate[0]*RelativeCoordinate[1])*temp[nn]
            A[nn,4] = (RelativeCoordinate[0]*RelativeCoordinate[0])*temp[nn]
            A[nn,5] = (RelativeCoordinate[1]*RelativeCoordinate[1])*temp[nn]  
                    
            b_sens[nn] = Sensitivities[Indices[nn,0],Indices[nn,1]]*temp[nn]

        B = np.linalg.lstsq(A,b_sens)[0][0]
        
        if (B > Bmax*10) or (B < Bmin*10):
            B = 0.0
            temp = 0.
            for nn in range(0,CntPoints):
                Temp = self.CLinElasticity.CMesh.AreaFraction[Indices[nn,0]]
                B += Sensitivities[Indices[nn,1]]*Temp
                temp += Temp
            PointSensitivities = B/temp
        elif B > Bmax:
            PointSensitivities = Bmax
        elif B < Bmin:
            PointSensitivities = Bmin
        else:
            PointSensitivities = B
        
        return PointSensitivities

class ElasticitySensitivities(SensitivityAnalysis):
    def __init__(self,CLinElasticity):
        self.AllowedAreaFraction = 0.001
        super(ElasticitySensitivities,self).__init__(CLinElasticity,self.AllowedAreaFraction)
       
    def Compliance(self,BoundaryPoints,Weights,Radius,WeightFlag,Tolerance):
        # calling function of ComputeBoundaryPointSensitivities()
        self.IntegrationPointFieldGradients(Tolerance)
        # Sensitivities = np.zeros(BoundaryPoints.shape[0]) # output1
        BoundarySensitivities = np.zeros(BoundaryPoints.shape[0]) # output2
        IntegrationPointSensitivties \
            = np.zeros((self.CLinElasticity.CMesh.nELEM,self.CLinElasticity.NumGpts**2))
        
        for ee in range(0,self.CLinElasticity.CMesh.nELEM):
            if self.CLinElasticity.CMesh.AreaFraction[ee] > Tolerance :
                for gg in range(0,self.CLinElasticity.NumGpts**2):
                    TempStress = self.CLinElasticity.Cijkl.dot(self.FieldGradient[ee,gg,:])
                    TempStress *= (self.CLinElasticity.CMesh.AreaFraction[ee])
                    
                    IntegrationPointSensitivties[ee,gg] = TempStress.dot(self.FieldGradient[ee,gg,:])*Weights #VERIFIED
                    
        
        for ss in range(0,BoundaryPoints.shape[0]): 
            BoundarySensitivities[ss] = self.ComputeBoundaryPointSensitivities(BoundaryPoints[ss],IntegrationPointSensitivties, Radius, WeightFlag, Tolerance)
            # BoundarySensitivities[ss] = Sensitivities[ss]
        
        return IntegrationPointSensitivties, BoundarySensitivities
