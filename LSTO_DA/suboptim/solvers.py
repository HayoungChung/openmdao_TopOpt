import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
from pylab import *
from scipy.optimize import linprog

class Solvers(object):
    # def __init__(self,bpts_xy: (np.ndarray), Sf: (np.ndarray), Sg: (np.ndarray), Cf: (np.ndarray), Cg: (np.ndarray), length_x: (float), length_y: (float), areafraction: (np.ndarray), movelimit:(float) = 0.2, isprint = False):
    def __init__(self,bpts_xy, Sf, Sg, Cf, Cg, length_x, length_y, areafraction, movelimit = 0.2, isprint = False):
        self.bpts_xy = bpts_xy
        self.Sf = Sf
        self.Sg = Sg
        self.Cf = Cf
        self.Cg = Cg
        self.length_x = length_x
        self.length_y = length_y
        self.areafraction = areafraction
        self.movelimit = movelimit

        self.nBpts = int(bpts_xy.shape[0])


    def simplex(self, isprint=False):
        bpts_xy = self.bpts_xy
        Sf = self.Sf
        Sg = self.Sg
        Cf = self.Cf
        Cg = self.Cg
        length_x = self.length_x
        length_y = self.length_y
        areafraction = self.areafraction
        movelimit = self.movelimit
        nBpts = self.nBpts

        percent_area = 0.5
        target_total = 0.4
        target_area0 = sum(areafraction)
        target_area = target_area0

        # target area
        for ii in range(nBpts):
            target_area += Cg[ii] * percent_area * (movelimit)
        target_area = max(target_total * length_x * length_y, target_area)

        if (isprint):
            np.savetxt("areas_constraint.txt", np.array([target_area0, target_area]))
            print("target = ")
            print(target_area / length_x/ length_y)

        # distance vector
        bounds = [[None, None]]*nBpts
        bounds_np = np.ndarray((nBpts,2),dtype=float)

        domain_distance_vector = np.zeros(nBpts)
        for ii in range(nBpts):
            px_ = bpts_xy[ii,0]
            py_ = bpts_xy[ii,1]
            # assume rectangular design domain
            domdist = min([abs(px_ -0.0), abs(px_ - length_x), abs(py_ - length_y), abs(py_ - 0.0)])
            # if ( (px_ >= length_x) or ( px_ <= 0.0) or (py_ >= length_y) or (py_ <= 0.0) ):
            #     domdist = -1.0 * domdist # NB: neccesary?
            domain_distance_vector[ii] = -np.min([domdist, movelimit])

            bounds[ii][0] = np.min([domain_distance_vector[ii], movelimit])
            bounds[ii][1] = movelimit
            bounds_np[ii,0]  = float(bounds[ii][0])
            bounds_np[ii,1]  = float(bounds[ii][1])

        # SIMPLEX matrices
        Cmat = Cf
        A_eq = np.ndarray((1,nBpts), dtype=float)
        A_eq[0] = Cg
        b_eq = np.ndarray((1), dtype=float)
        b_eq[0] = target_area - sum(areafraction)

        if (isprint):
            print ([Cf.dot(bounds_np[:,0]),Cf.dot(bounds_np[:,1]), Cg.dot(bounds_np[:,0]), Cg.dot(bounds_np[:,1]), b_eq])
        xout = scipy.optimize.linprog(method='interior-point', c=Cmat, A_eq=A_eq, b_eq=b_eq, bounds=bounds_np) #, options={"disp": True})
        Z_bpts = xout.x

        new_area = sum(areafraction)
        new_area += Cg.dot(Z_bpts)

        if (isprint):
            print("original area = %1.4f, target_area = %1.4f, new_area = %1.4f" % (sum(areafraction), target_area, new_area))  # validated

        # velocity calculation
        # Bpt_Vel = np.zeros(nBpts)
        # for ii in range(nBpts):
        #     domdist = domain_distance_vector[ii]
        #     Bpt_Vel[ii] = -1.0*min(Z_bpts[ii], domdist)

        Bpt_Vel = Z_bpts/1.0
        if (isprint):
            clf(); scatter(bpts_xy[:,0],bpts_xy[:,1],c=Bpt_Vel); colorbar(); savefig("a.png")

        abs_Vel = np.amax(np.abs(Bpt_Vel))

        if (abs_Vel > movelimit):
            Bpt_Vel *= movelimit/abs_Vel

        return Bpt_Vel

    def bisection(self, isprint=False):
        bpts_xy = self.bpts_xy
        Sf = self.Sf
        Sg = self.Sg
        Cf = self.Cf
        Cg = self.Cg
        length_x = self.length_x
        length_y = self.length_y
        areafraction = self.areafraction
        movelimit = self.movelimit
        nBpts = self.nBpts

        percent_area = 0.5
        target_total = 0.5
        target_area = sum(areafraction)

        # target area
        for ii in range(nBpts):
            target_area += Cg[ii] * percent_area * (-movelimit)

        target_area = max(target_total * length_x * length_y, target_area)

        if (isprint):
            print("target = ")
            print(target_area / length_x/ length_y)
        # bisection method assuming rectangular design domain

        # distance vector =====
        domain_distance_vector = np.zeros(nBpts)
        for ii in range(nBpts):
            px_ = bpts_xy[ii,0]
            py_ = bpts_xy[ii,1]

            domdist = min([abs(px_ -0.0), abs(px_ - length_x), abs(py_ - length_y), abs(py_ - 0.0)])
            if ( (px_ >= length_x) or ( px_ <= 0.0) or (py_ >= length_y) or (py_ <= 0.0) ):
                domdist = -1.0 * domdist
            domain_distance_vector[ii] = min(domdist, movelimit)

        lambda_0 = 0.0 # default parameter
        default_area = sum(areafraction)
        for ii in range(nBpts):
            default_area += Cg[ii]*min(domain_distance_vector[ii], movelimit*Sg[ii] + lambda_0*Sf[ii])


        delta_lambda = 0.1 # perturbation
        for iITER in range(100):
            if (iITER == 99):
                print("bisection failed")
                print(new_area0/length_x/length_y, target_area/length_x/length_y)

            lambda_curr = lambda_0
            new_area0 = sum(areafraction)
            for kk in range(nBpts):
                new_area0 += Cg[kk]*min( domain_distance_vector[kk], movelimit*Sg[kk] + lambda_curr*Sf[kk] )

            lambda_curr = lambda_0 + delta_lambda
            new_area2 = sum(areafraction)
            for kk in range(nBpts):
                new_area2 += Cg[kk]*min( domain_distance_vector[kk], movelimit*Sg[kk] + lambda_curr*Sf[kk] )

            lambda_curr = lambda_0 - delta_lambda
            new_area1 = sum(areafraction)
            for kk in range(nBpts):
                new_area1 += Cg[kk]*min( domain_distance_vector[kk], movelimit*Sg[kk] + lambda_curr*Sf[kk] )

            slope = (new_area2 - new_area1) / 2 / delta_lambda

            lambda_0 -= (new_area0 - target_area) / slope

            if (isprint):
                print(" new_area2 = ", new_area2, " new_area1 = ", new_area1)
                print(" lambda_f = ", lambda_0, " target_area = ", target_area, " new_area0 = ", new_area0)

            # termination
            if (abs(new_area0 - target_area) < 1.0E-3):
                print([new_area0/length_x/length_y, target_area/length_x/length_y])
                break

            # iteration fin

        lambda_f = lambda_0

        # velocity calculation
        Bpt_Vel = np.zeros(nBpts)
        for ii in range(nBpts):
            domdist = domain_distance_vector[ii]
            Bpt_Vel[ii] = -1.0*min( lambda_f*Sf[ii] + movelimit*Sg[ii], domdist)

        abs_Vel = np.amax(np.abs(Bpt_Vel))

        if (isprint):
            clf(); scatter(bpts_xy[:,0],bpts_xy[:,1],c=Bpt_Vel); colorbar(); savefig("a.png")

        if (abs_Vel > movelimit):
            Bpt_Vel *= movelimit/abs_Vel

        return Bpt_Vel # boundary point velocity
