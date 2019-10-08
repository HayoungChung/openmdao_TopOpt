# thermoelastic

from pylab import *
from openmdao.api import ExplicitComponent, ImplicitComponent

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

try:
    import cPickle as pickle
except:
    import pickle

from pyBind import py_FEA

class HEAT_FE(object):
    @staticmethod
    def _getCijkl(E, nu):
        Cijkl = np.zeros([3,3])
        Cijkl = np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., 0.5*(1.-nu)]])
        Cijkl *= E/(1.-nu*nu)
        return Cijkl

    @staticmethod
    def _getNmat():
        Nmat = 0.25*np.array([1., 1., 1., 1.])
        return Nmat

    @staticmethod
    def _getBmat(lxy, nelxy):

        dNdX = HEAT_FE._getdNdX(lxy, nelxy, 0.0, 0.0)

        Bmat = np.zeros([3,8], dtype=float)

        dN_dx = dNdX[0,:]
        dN_dy = dNdX[1,:]

        Bmat[0,0::2] = dN_dx
        Bmat[1,1::2] = dN_dy
        Bmat[2,0::2] = dN_dy
        Bmat[2,1::2] = dN_dx

        return Bmat

    @staticmethod
    def _getdNdX(lxy, nelxy, r, s):
        # NB: works only in structured Q4 mesh, of which grid is coaxial with x-y grid
        drdx = 2.0 * nelxy[0]/lxy[0]
        dsdy = 2.0 * nelxy[1]/lxy[1]

        dNdX = np.zeros([2,4], dtype=float)
        dN_dx = 0.25 * drdx * np.array([-(1.-s), (1.-s), (1.+s), -(1.+s)])
        dN_dy = 0.25 * dsdy * np.array([-(1.-r), -(1.+r), (1.+r), (1.-r)])

        dNdX[0,:] = dN_dx
        dNdX[1,:] = dN_dy

        return dNdX

class ConductComp(ImplicitComponent):
    def initialize(self):
        self.options.declare('k_cond', types=(float))
        self.options.declare('nelx', types=(int))
        self.options.declare('nely', types=(int))
        self.options.declare('length_x', types=(float))
        self.options.declare('length_y', types=(float))
        self.options.declare('ELEM', types=np.ndarray)
        self.options.declare('BCid', types=np.ndarray)

    def setup(self):
        self.k_cond = k_cond = self.options['k_cond'] # conductivity
        self.nelx = nelx = float(self.options['nelx'])
        self.nely = nely = float(self.options['nely'])
        self.length_x = length_x = self.options['length_x']
        self.length_y = length_y = self.options['length_y']
        self.ELEM = ELEM = self.options['ELEM']
        self.BCid = BCid = self.options['BCid']

        # inputs and outputs
        self.nELEM = nELEM = int(nelx*nely)
        self.nNODE = nNODE = int((nelx+1)*(nely+1))
        self.nDOF = nDOF = nNODE
        self.nDOF_withLambda = nDOF_withLambda = nDOF + len(BCid)
        self.add_input('rhs', shape=nDOF_withLambda) # generation of heat at the each element
        self.add_input('multipliers', shape=nELEM) # elem-wise material density
        self.add_output('states', shape=nDOF_withLambda) # temperarature (elem-wise)

        # matrix properties
        self.detJ = (length_x*length_y) /(nelx*nely) / 4 # determinant of ||dxy/drs||

        # declare partials
        nBCid = len(BCid)
        idx = np.zeros(nELEM*16 + nBCid*2, dtype=int)
        idy = np.zeros(nELEM*16 + nBCid*2, dtype=int)
        idx_d = np.zeros(nELEM*4, dtype=int)
        idy_d = np.zeros(nELEM*4, dtype=int)
        for ee in range(nELEM):
            ELEM_id = np.kron(ELEM[ee],np.ones([4,1]))
            idx[16*ee:16*(ee+1)] = ELEM_id.flatten()
            idy[16*ee:16*(ee+1)] = ELEM_id.transpose().flatten()

            ELEM_id = ELEM[ee]
            idx_d[4*ee:4*(ee+1)] = ELEM_id
            idy_d[4*ee:4*(ee+1)] = [ee, ee, ee, ee]

        # BC apply
        idx[16*self.nELEM:16*self.nELEM+nBCid] = np.arange(self.nDOF,self.nDOF+nBCid)
        idy[16*self.nELEM:16*self.nELEM+nBCid] = self.BCid
        idy[16*self.nELEM+nBCid::] = np.arange(self.nDOF,self.nDOF+nBCid)
        idx[16*self.nELEM+nBCid::] = self.BCid

        self.declare_partials('states', 'states', rows=idx, cols=idy)
        self.declare_partials('states', 'multipliers', rows=idx_d , cols=idy_d)

        data = np.zeros(nDOF_withLambda)
        data[:nDOF_withLambda] = -1.0
        arange = np.arange(nDOF_withLambda)
        self.declare_partials('states', 'rhs', rows=arange, cols=arange, val=data)

    def _KEL(self):
        ri = [-1./np.sqrt(3.),+1./np.sqrt(3.),+1./np.sqrt(3.), -1./np.sqrt(3.)]
        si = [-1./np.sqrt(3.),-1./np.sqrt(3.),+1./np.sqrt(3.), +1./np.sqrt(3.)]
        KEL = np.zeros([4,4], dtype=float)
        for gg in range(4):
            r = ri[gg]
            s = si[gg]
            w = 1.0
            Bmat = HEAT_FE._getdNdX([self.length_x, self.length_y], [self.nelx, self.nely], r, s) # NB: Bmat evaluated at the centriod
            KEL += Bmat.transpose().dot(Bmat) * self.detJ * w

        return KEL

    def _getmtx(self, inputs):
        multiplier = inputs['multipliers']
        nBCid = len(self.BCid)

        # compute sparse mtx
        KE0 = self._KEL()
        idx = np.zeros(self.nELEM*16 + nBCid*2, dtype=int)
        idy = np.zeros(self.nELEM*16 + nBCid*2, dtype=int)
        val = np.zeros(self.nELEM*16 + nBCid*2)
        for ee in range(self.nELEM):
            k_curr = self.k_cond * multiplier[ee] + (self.k_cond*1e-9)*(1-multiplier[ee])
            KE_tmp = KE0 * k_curr #multiplier[ee] + (1-multiplier[ee])

            ELEM_id = np.kron(self.ELEM[ee],np.ones([4,1]))
            idx[16*ee:16*(ee+1)] = ELEM_id.flatten()
            idy[16*ee:16*(ee+1)] = ELEM_id.transpose().flatten()
            val[16*ee:16*(ee+1)] = KE_tmp.flatten()

        # BC apply
        idx[16*self.nELEM:16*self.nELEM+nBCid] = np.arange(self.nDOF,self.nDOF+nBCid)
        idy[16*self.nELEM:16*self.nELEM+nBCid] = self.BCid
        idy[16*self.nELEM+nBCid::] = np.arange(self.nDOF,self.nDOF+nBCid)
        idx[16*self.nELEM+nBCid::] = self.BCid
        val[16*self.nELEM::] = 1.0

        npval = np.array(val)
        npidx = np.array(idx, dtype=np.int32)
        npidy = np.array(idy, dtype=np.int32)
        mtx = sp.sparse.csc_matrix((npval, (npidx, npidy)),
        shape=(self.nNODE+nBCid, self.nNODE+nBCid))
        return (mtx, npval)

    def _getmtx_deriv(self, outputs):
        temp = outputs['states']
        # compute sparse mtx
        KE0 = self._KEL()
        # idx = idy = np.zeros(self.nELEM*4, dtype=int)
        val = np.zeros(self.nELEM*4)

        #### TEMP_PLOT ####
        # checking temperatures distribution
        # SHOULD BE REMOVED AFTER IT IS CHECKED
        # xlin = np.linspace(0,160,161)
        # ylin = np.linspace(0,80,81)
        # xx, yy = np.meshgrid(xlin,ylin)
        # temp_re = np.reshape(temp[:self.nDOF],[81,161])
        # contour(xx,yy,temp_re)
        # axis('equal')
        # colorbar()
        # grid(True)
        # box
        # savefig('temperatures_contour.png')
        # clf()
        # exit()
        #### TEMP_PLOT ####
        for ee in range(self.nELEM):
            elem_id = self.ELEM[ee]
            temp_el = temp[elem_id]
            KE_curr = KE0 * (self.k_cond*(1.0))
            KU = KE0.dot(temp_el)

            # ELEM_id = self.ELEM[ee]
            # idx[4*ee:4*(ee+1)] = ELEM_id.flatten()
            # idy[4*ee:4*(ee+1)] = ee
            val[4*ee:4*(ee+1)] = KU.flatten()

        npval = np.array(val)
        # npidx = np.array(idx, dtype=np.int32)
        # npidy = np.array(idy, dtype=np.int32)
        return npval

    def apply_nonlinear(self, inputs, outputs, residuals):
        mtx = self._getmtx(inputs)[0]
        residuals['states'] = mtx.dot(outputs['states']) - inputs['rhs']

    def solve_nonlinear(self, inputs, outputs):
        mtx = self._getmtx(inputs)[0]
        outputs['states'] = sp.sparse.linalg.spsolve(mtx, inputs['rhs']) # direct solver: 2D

    def linearize(self, inputs, outputs, partials):
        (self.mtx, K_total) = self._getmtx(inputs)
        partials['states', 'states'] = K_total

        Kij_deriv = self._getmtx_deriv(outputs)
        partials['states', 'multipliers'] = Kij_deriv

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            #self._solve(d_outputs['states'], d_residuals['states'], 'fwd')
            d_outputs['states']= scipy.sparse.linalg.spsolve(self.mtx, d_residuals['states'])

        elif mode == 'rev':
            #self._solve(d_residuals['states'], d_outputs['states'], 'rev')
            d_residuals['states']= scipy.sparse.linalg.spsolve(self.mtx, d_outputs['states'])
        # pass


class ThermCoupleLoadComp(ExplicitComponent):
    # this computes thermal loads (F_thermal = int_\Omega B^T:C:\epsilon_th)
    def initialize(self):
        self.options.declare('alpha', types=(float)) # thermal expansion coeff
        self.options.declare('nelx', types=(int))
        self.options.declare('nely', types=(int))
        self.options.declare('length_x', types=(float))
        self.options.declare('length_y', types=(float))
        self.options.declare('ELEM', types=np.ndarray)
        self.options.declare('DOF_m2do', types=np.ndarray) # stupid numbering
        self.options.declare('E', types=float)
        self.options.declare('nu', types=float)
        self.options.declare('BCid', types= np.ndarray) # mechanical dof

    def setup(self):
        self.alpha = alpha = self.options['alpha']
        self.nelx = nelx = float(self.options['nelx'])
        self.nely = nely = float(self.options['nely'])
        self.length_x = length_x = self.options['length_x']
        self.length_y = length_y = self.options['length_y']
        self.ELEM = ELEM = self.options['ELEM']
        self.E = E = self.options['E']
        self.nu = nu = self.options['nu']
        self.BCid = BCid = self.options['BCid']
        self.DOF_m2do = DOF_m2do = self.options['DOF_m2do']

        # inputs and outputs
        self.nELEM = nELEM = int(nelx*nely)
        self.nNODE = nNODE = int((nelx+1)*(nely+1))
        self.nDOF = nDOF = nNODE
        self.nDOF_wLambda = nDOF_wLambda = nNODE*2+len(BCid)
        self.add_input('temperatures', shape=nDOF) # nodal temperatures
        self.add_input('multipliers', shape=nELEM) # elem-wise density
        self.add_input('f_m', shape=nDOF_wLambda)
        self.add_output('f_tot', shape=nDOF_wLambda)

        # variables that are constant throughout design
        self.Bmat = HEAT_FE._getBmat([length_x, length_y], [nelx, nely])
        self.Nmat = HEAT_FE._getNmat()
        self.Cijkl0 = HEAT_FE._getCijkl(E, nu)

        # declare of the partials
        idx_dfdt = np.zeros([nELEM*(8*4)], dtype=int)
        idy_dfdt = np.zeros([nELEM*(8*4)], dtype=int)
        idx_dfdr = np.zeros([nELEM*(8)], dtype=int)
        idy_dfdr = np.zeros([nELEM*(8)], dtype=int)
        for ee in range(nELEM):
            elem_id = ELEM[ee]
            elem_dof = DOF_m2do[ee]

            # partial df_dT
            id_kron = np.kron(elem_dof,np.ones([4,1]))
            idx_dfdt[ee*8*4:(ee+1)*8*4] = id_kron.flatten()
            id_kron = np.kron(elem_id,np.ones([8,1])).transpose()
            idy_dfdt[ee*8*4:(ee+1)*8*4] = id_kron.flatten()

            # partial df_drho
            idx_dfdr[ee*8:(ee+1)*8] = elem_dof.flatten()
            idy_dfdr[ee*8:(ee+1)*8] = [ee] * 8

        self.declare_partials(of='f_tot', wrt='temperatures', rows=idx_dfdt, cols=idy_dfdt)
        self.declare_partials(of='f_tot', wrt='multipliers', rows=idx_dfdr, cols=idy_dfdr)
        self.declare_partials(of='f_tot', wrt='f_m', rows=arange(nDOF_wLambda), cols=arange(nDOF_wLambda), val=np.ones(nDOF_wLambda))

    def compute(self, inputs, outputs):
        f_th = np.zeros(self.nDOF*2+len(self.BCid))

        T_dof = np.array(inputs['temperatures'])
        rho_elem = np.array(inputs['multipliers'])

        wi_red = 4.0 # wieghting (r=s=0)

        for ee in range(self.nELEM):
            elem_id = self.ELEM[ee]
            elem_dof = self.DOF_m2do[ee]

            # evaluate deltaT at the centroid of element
            T_elem = T_dof[elem_id]
            T_center = self.Nmat.dot(T_elem) # same as taking average
            alphaT = self.alpha * T_center
            epsilon_T = (alphaT) * np.array([1., 1., 0.]).transpose()

            # thermal stress
            sigma_T = self.Cijkl0.dot(epsilon_T)*rho_elem[ee]

            # thermal force
            f_loc = self.Bmat.transpose().dot(sigma_T) # local force
            f_loc *= wi_red * (self.length_x * self.length_y) / (self.nelx*self.nely)/ 4.
            f_th[elem_dof] += f_loc

        f_th[self.BCid] = 0.0
        outputs['f_tot'] = f_th + inputs['f_m']

    def compute_partials(self, inputs, partials):
        T_dof = np.array(inputs['temperatures'])
        rho_elem = np.array(inputs['multipliers'])

        wi_red = 4.0 # wieghting (r=s=0)
        val_dfdt = np.zeros(self.nELEM * 8 * 4) # w.r.t. nodal temperatures
        val_dfdr = np.zeros(self.nELEM * 8) # w.r.t. element density

        for ee in range(self.nELEM):
            elem_id = self.ELEM[ee]

            # df_dT ==========================================================
            # evaluate deltaT at the centroid of element
            dfdt_loc = np.zeros(8*4)
            for tt in range(4):
                T_del = np.zeros(4, dtype = float)
                if (any(elem_id[tt]==self.BCid)):
                    T_del[tt]  = 0.0
                else:
                    T_del[tt]  = 1.0
                T_center = self.Nmat.dot(T_del)
                alphaT = self.alpha * T_center
                epsilon_T = alphaT * np.array([1., 1., 0.]).transpose()

                # thermal stress
                sigma_T = self.Cijkl0.dot(epsilon_T) * rho_elem[ee]

                dfdt_loc[8*tt:8*(tt+1)] = self.Bmat.transpose().dot(sigma_T) * wi_red * (self.length_x * self.length_y) / (self.nelx*self.nely)/ 4. # local
            val_dfdt[ee*8*4:(ee+1)*8*4] = dfdt_loc.flatten()

            # df_drho ========================================================
            # evaluate deltaT at the centroid of element
            T_elem = T_dof[elem_id]
            T_center = self.Nmat.dot(T_elem) # same as taking the average
            alphaT = self.alpha * T_center
            epsilon_T = (alphaT) * np.array([1., 1., 0.]).transpose()

            # thermal stress
            sigma_T = self.Cijkl0.dot(epsilon_T)

            dfdr_loc = self.Bmat.transpose().dot(sigma_T) * wi_red * (self.length_x * self.length_y) / (self.nelx*self.nely)/ 4. # local
            val_dfdr[ee*8:(ee+1)*8] = dfdr_loc.flatten()

        partials['f_tot', 'temperatures'] = val_dfdt
        partials['f_tot', 'multipliers'] = val_dfdr


