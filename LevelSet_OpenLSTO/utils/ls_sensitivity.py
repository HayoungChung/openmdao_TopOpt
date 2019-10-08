# sensitivity analysis for least square method

import numpy as np

class _LeastSquare(object):
    def __init__(self, bpts_xy_, gpts_xy_, gpts_sens_, area_fraction_, radius = 2):
        
        self.bpts_xy = bpts_xy_
        self.WeightFlag = 5 # default
        self.Weights = 1
        self.Radius = radius
        self.AllowedAreaFraction = 0.01      

        self.gpts_xy = gpts_xy_             #[num_elem, order_gpts**2, 2]
        self.gpts_sens = gpts_sens_         #[num_elem, order_gpts**2, 2]
        self.area_fraction = area_fraction_

        self.num_bpts = self.bpts_xy.shape[0]
        self.num_gpts = self.gpts_xy.shape[0]
        self.num_elem = self.area_fraction.shape[0]
        self.gpts_elem = self.gpts_xy.shape[1]  # gpts per elem

    def get_sens_compliance(self):
        BoundarySensitivities = np.zeros(self.num_bpts)
        for ss in range(0,self.num_bpts): 
            BoundarySensitivities[ss] = self.ComputeBoundaryPointSensitivities(\
                self.bpts_xy[ss],
                self.gpts_xy,
                self.gpts_sens, 
                self.area_fraction,
                self.Radius, 
                self.WeightFlag)
             
        return BoundarySensitivities

    def ComputeBoundaryPointSensitivities(self,
                BoundaryPoints, 
                fixedGpts_xy, Sensitivities,
                area_fraction, 
                Radius = 2.0, Weightflag = 5, Tolerance = 0.001):

        p = int( ((3.14*Radius**2) / 1 ) * 4 * (5./4.) )  # conservative measure
        Distances = np.zeros(p)
        Indices = np.zeros((p,2)).astype(int)
        PointSensitivities = 0.
        
        CntPoints = int(0)

        el_cood = np.average(fixedGpts_xy, axis = 1)
        el_dist = np.sqrt(np.sum( ( np.tile(BoundaryPoints,(self.num_elem,1)) - el_cood )**2, axis=1)) 
        id_e_within = np.nonzero(el_dist < 1.5*Radius)[0]
        ee_tmp_ = 0
        CntPoints = 0
        for ee in id_e_within:
            gg_dist = np.sqrt(np.sum((np.tile(BoundaryPoints,(self.gpts_elem,1))-fixedGpts_xy[ee,:,:])**2,axis = 1))
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
            temp = area_fraction[Indices[0:CntPoints,0]]
        elif Weightflag == 4:
            temp = area_fraction[Indices[0:CntPoints,0]]/Distances[0:CntPoints]
        elif Weightflag == 5:
            temp = np.sqrt(area_fraction[Indices[0:CntPoints,0]]/Distances[0:CntPoints])
        else:
            temp = 1
            print("Weight Flag should lie in [1, 5]. Using Least Squares.\n") 

        for nn in range(0,CntPoints):
            RelativeCoordinate = fixedGpts_xy[Indices[nn,0],Indices[nn,1]] - BoundaryPoints
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
                Temp = area_fraction[Indices[nn,0]]
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

