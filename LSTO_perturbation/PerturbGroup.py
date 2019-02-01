
from openmdao.api import Group, Problem, pyOptSparseDriver, view_model
from openmdao.api import IndepVarComp, ExplicitComponent, ImplicitComponent
from openmdao.api import DirectSolver

import sys
sys.path.append('../Density_OpenLSTO')
sys.path.append('../LevelSet_OpenLSTO')

from py_lsmBind import py_LSM 
from pyBind import py_FEA, py_Sensitivity

from pylab import *

# from SIMP components
from components.states_comp import StatesComp as SIMP_StatesComp # R = KU-F
from components.compliance_comp import ComplianceComp as SIMP_ComplianceComp
from components.disp_comp import DispComp as SIMP_DispComp
from components.weight_comp import WeightComp as SIMP_WeightComp

# from LSTO components
from components_new.ConstraintComp import ConstraintComp as LSTO_Constraint
from components_new.DisplacementComp import DisplacementComp as LSTO_DisplacementComp
from components_new.IntegralComp import IntegralComp as LSTO_IntegralComp
from components_new.ObjectiveComp import ObjectiveComp as LSTO_ObjectiveComp
from components_new.ScalingComp  import ScalingComp as LSTO_ScalingComp

# from perturbation
from DiscretizeComp import DiscretizeComp
# from stress_comp import MaxStressComp

import scipy.sparse
import scipy.sparse.linalg

class PerturbGroup(Group):

    def initialize(self):
        self.options.declare('fea_solver', types=py_FEA)
        self.options.declare('lsm_solver',types=py_LSM )
        self.options.declare('nelx', types=int)
        self.options.declare('nely', types=int)
        self.options.declare('force', types= ndarray)
        # self.options.declare('movelimit', types= float)
    
    def setup(self):
        self.lsm_solver = lsm_solver = self.options['lsm_solver']
        self.fea_solver = fea_solver = self.options['fea_solver']
        self.force = force = self.options['force']
        self.nelx = nelx = self.options['nelx']
        self.nely = nely = self.options['nely']
        # self.movelimit = movelimit = self.options['movelimit']

        phi = lsm_solver.get_phi()
        nELEM = self.nELEM = nelx*nely
        nNODE = self.nNODE = (1+nelx) * (1+nely)
        nDOF = nNODE * 2

        (bpts_xy, areafraction, segLength) = lsm_solver.discretise()
        nBpts = self.nBpts = bpts_xy.shape[0]
        '''
        IMPORTANT: CHECKING ERRORS

        num_dofs_w_lambda = nDOF + 162

        (rows, cols, vals) = fea_solver.compute_K_SIMP(areafraction)
        nprows = np.array(rows, dtype=np.int32)
        npcols = np.array(cols, dtype=np.int32)
        npvals = np.array(vals, dtype=float)
        K_sparse = scipy.sparse.csc_matrix((npvals, (nprows,npcols))    , 
                                shape=(num_dofs_w_lambda,num_dofs_w_lambda))

        coord = np.array([160,40])
        tol = np.array([1e-3, 1e-3]) 
        GF_ = fea_solver.set_force(coord = coord,tol = tol, direction = 1, f = -1.0)
        GF = np.zeros(num_dofs_w_lambda)
        GF[:nDOF] = GF_
        u = scipy.sparse.linalg.spsolve(K_sparse, GF)[:nDOF]

        print(min(u))
        print(sum(u))
        print(np.linalg.norm(u))
        exit(0)
        '''

        # 0. boundary condition setup (via Lagrange multiplier)
        (nodes, elem, elem_dof) = fea_solver.get_mesh()
        length_x = max(nodes[:, 0])
        length_y = max(nodes[:, 1])
        
        # SIMP_1. get boundary points
        comp_ = IndepVarComp()
        comp_.add_output('rhs', val = force)
        comp_.add_output('bpts', val = bpts_xy)
        self.add_subsystem('inputs_comp', comp_)
        self.connect('inputs_comp.bpts', 'area_comp.points')
        self.connect('inputs_comp.rhs', 'states_comp.rhs')
        self.connect('inputs_comp.rhs', 'GF_comp.states')

        # SIMP_2. boundary-to-area
        comp_ = DiscretizeComp(lsm_solver=lsm_solver, nelx=nelx, nely=nely, 
                                    nBpts=nBpts, perturb=0.2)
        self.add_subsystem('area_comp', comp_)
        self.connect('area_comp.density', 'states_comp.multipliers')
        self.connect('area_comp.density', 'weight_comp.x')

        # SIMP_3. area-to-state
        comp_ = SIMP_StatesComp(fem_solver=fea_solver, num_nodes_x=nelx+1, num_nodes_y=nely+1, isSIMP=True)
        self.add_subsystem('states_comp', comp_)
        self.connect('states_comp.states', 'disp_comp.states')

        # SIMP_4. extract out lagrange multipliers
        comp_ = SIMP_DispComp(num_nodes_x = nelx+1, num_nodes_y = nely+1)
        self.add_subsystem('disp_comp', comp_)
        self.connect('disp_comp.disp', 'compliance_comp.disp')

        comp_ = SIMP_DispComp(num_nodes_x = nelx+1, num_nodes_y = nely+1)
        self.add_subsystem('GF_comp', comp_)
        self.connect('GF_comp.disp', 'compliance_comp.forces')

        # SIMP_5. compliance
        comp_ = SIMP_ComplianceComp(num_nodes_x = nelx+1, num_nodes_y = nely+1)
        self.add_subsystem('compliance_comp', comp_)
        self.add_objective('compliance_comp.compliance')

        # # SIMP_5. maximum von mises
        # comp_ = MaxStressComp(nelx = nelx, nely = nely, fea_solver=fea_solver)
        # self.add_subsystem('compliance_comp', comp_)
        # self.add_objective('compliance_comp.maxStress')

        # SIMP_6. total area
        comp_ = SIMP_WeightComp(num=nELEM)
        self.add_subsystem('weight_comp', comp_)
        self.add_constraint('weight_comp.weight', upper = 0.4)


        # fixme: solve_linear() is called so many times 
        # dummy
        self.add_design_var('inputs_comp.bpts') # without this, total derivative is not calculated
        

        # self.options['assembled_jac_type'] = 'csc'
        # self.linear_solver = DirectSolver(assemble_jac=True)