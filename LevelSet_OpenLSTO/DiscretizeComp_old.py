# creating signed distance field based on the point cloud

from openmdao.api import ExplicitComponent
from pylab import *

from py_lsmBind import py_LSM 

class DiscretizeComp(ExplicitComponent):
    def initialize(self):
        self.options.declare(name = 'lsm_solver', types=py_LSM)
        self.options.declare(name = 'nelx', types = int)
        self.options.declare(name = 'nely', types = int)
        self.options.declare(name = 'nBpts', types = int)
        self.options.declare(name = 'perturb', types= float)
        
    def setup(self):
        self.lsm_solver = self.options['lsm_solver']
        nelx = self.nelx = self.options['nelx']
        nely = self.nely = self.options['nely']
        self.nELEM = nELEM = nelx * nely
        self.nBpts = nBpts = self.options['nBpts']
        self.pertb = self.options['perturb']
        self.add_input('points', shape=[nBpts, 2])
        self.add_output('density', shape=nELEM)
        self.cnt_tmp = 0

    def compute(self, inputs, outputs):
        print(self.cnt_tmp)
        self.cnt_tmp += 1
        (bpts_xy, areafraction, seglength) = self.lsm_solver.discretise()
        self.seglength = seglength
        outputs['density'] = areafraction

        # computing partials and declare: sparse partials
        bpts_xy = inputs['points']
        
        (row, col, val) = self._get_pertb_partials_(bpts_xy)
        self.declare_partials('density', 'points', rows = row, cols = col, val = val)

    
    def _get_pertb_partials_(self, bpts_xy):
        lsm_solver = self.lsm_solver
        nelx = self.nelx
        nely = self.nely
        nELEM = self.nELEM
        nBpts = self.nBpts
        pertb = self.pertb
        mat_ = zeros((nELEM, nBpts),order='F')
                
        row = []
        col = [] 
        val = []

        for bbb in range(0, nBpts):
            px_ = bpts_xy[bbb,0]
            py_ = bpts_xy[bbb,1]

            nelx_pert_0 = int(max(int(floor(px_)) - 1 - 3, 0))
            nelx_pert_1 = int(min(int(floor(px_ - 1e-4)) + 2 + 3, int(nelx)))

            nely_pert_0 = int(max(int(floor(py_)) - 1 - 3, 0))
            nely_pert_1 = int(min(int(floor(py_ - 1e-4)) + 2 + 3, int(nely)))

            # dimensions of perturbed mesh
            nelx_pert = nelx_pert_1 - nelx_pert_0
            nely_pert = nely_pert_1 - nely_pert_0

            # level-set perturbation mesh
            lsm_pert = py_LSM(nelx = nelx_pert, nely = nely_pert, moveLimit = 0.5)
            lsm_pert.add_holes(locx = [], locy = [], radius = [])
            lsm_pert.set_levelset()

            # assign appropriate signed distance values to the perturbed mesh
            phi_org = lsm_solver.get_phi()

            count_pert = 0
            for iy in range(0, nely_pert+1):
                for ix in range(0, nelx_pert+1):
                    global_x = nelx_pert_0 + ix
                    global_y = nely_pert_0 + iy

                    lsm_pert.set_phi(index = count_pert, value = phi_org[(nelx + 1)*global_y + global_x], isReplace = True)
                    count_pert += 1 
        
            lsm_pert.reinitialise()
            (bpts_xy_pert0, areafraction_pert0, seglength_pert0) = lsm_pert.discretise()

            timestep_pert = 1.0 # deltaT for perturbation

            # assign perturbation velocity at the boundary point
            bpt_length = 0.0 
            vel_bpts = np.zeros(bpts_xy_pert0.shape[0])

            for ii in range(0, bpts_xy_pert0.shape[0]):
                tmp_px_ = bpts_xy_pert0[ii,0]
                tmp_py_ = bpts_xy_pert0[ii,1]
                dist_pert = pow(-tmp_px_ + px_ - nelx_pert_0, 2) + pow(-tmp_py_ + py_ - nely_pert_0, 2)
                dist_pert = dist_pert**0.5
                if (dist_pert < pertb):
                    vel_bpts[ii] = pertb * (1.0 - pow(dist_pert/pertb, 2.0))
                else:
                    vel_bpts[ii] = 0.0

            lsm_pert.advect_woWENO(vel_bpts, timestep_pert)

            # discretize again to get perturbed data
            (bpts_xy_pert1, areafraction_pert1, seglength_pert1) = lsm_pert.discretise()

            # loop through the elements in the narrow band
            count_pert = 0
            for iy in range(0, nely_pert):
                for ix in range(0, nelx_pert):
                    global_x = nelx_pert_0 + ix
                    global_y = nely_pert_0 + iy
                    global_index = (nelx)*global_y + global_x

                    delta_x = min(areafraction_pert0[count_pert] - areafraction_pert1[count_pert], 0.8*pertb)

                    if (delta_x > 1e-3 * pertb):
                        row = append(row, global_index)
                        col = append(col, bbb)
                        val = append(val, delta_x / self.seglength[bbb])

                    count_pert += 1
        
        return (row, col, val)
