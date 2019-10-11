from openmdao.api import Group, Problem, pyOptSparseDriver, view_model
from openmdao.api import IndepVarComp, ExplicitComponent, ImplicitComponent
from openmdao.api import DirectSolver

from py_lsmBind import py_LSM
from pyBind import py_FEA, py_Sensitivity

from pylab import *

# from SIMP components
from components.SIMP.states_comp import StatesComp as SIMP_StatesComp # R = KU-F
from components.SIMP.compliance_comp import ComplianceComp as SIMP_ComplianceComp
from components.SIMP.compliance_comp import HeatComplianceComp as COND_ComplianceComp
from components.SIMP.disp_comp import DispComp as SIMP_DispComp
from components.SIMP.disp_comp import DispComp_1dpn as COND_DispComp
from components.SIMP.weight_comp import WeightComp_real as SIMP_WeightComp

# from LSTO components
from components.LSTO.ConstraintComp import ConstraintComp as LSTO_Constraint
from components.LSTO.DisplacementComp import DisplacementComp as LSTO_DisplacementComp
from components.LSTO.IntegralComp import IntegralComp as LSTO_IntegralComp
from components.LSTO.ObjectiveComp import ObjectiveComp as LSTO_ObjectiveComp
from components.LSTO.ScalingComp  import ScalingComp as LSTO_ScalingComp

# from perturbation
from components.DiscretizeComp import VnPerturbComp # DiscretizeComp
# from stress_comp import MaxStressComp
from components.stress_comp import VMStressComp, pVmComp, pnormComp, BodyIntegComp
# Conductivity
from components.therm_el_comp import ConductComp, ThermCoupleLoadComp
from components.sum_comp import SumComp

import scipy.sparse
import scipy.sparse.linalg

class StressGroup(Group):
    def initialize(self):
        self.options.declare('fea_solver', types=py_FEA)
        self.options.declare('lsm_solver',types=py_LSM )
        self.options.declare('nelx', types=int)
        self.options.declare('nely', types=int)
        self.options.declare('force', types= ndarray)
        self.options.declare('movelimit', types= float)
        self.options.declare('pval', types= float)
        self.options.declare('E', types=float)
        self.options.declare('nu', types=float)
    def setup(self):
        self.lsm_solver = lsm_solver = self.options['lsm_solver']
        self.fea_solver = fea_solver = self.options['fea_solver']
        self.force = force = self.options['force']
        self.nelx = nelx = self.options['nelx']
        self.nely = nely = self.options['nely']
        self.movelimit = movelimit = self.options['movelimit']
        self.pval = pval = self.options['pval']
        E = self.E = self.options['E']
        nu = self.nu = self.options['nu']


        phi = lsm_solver.get_phi()
        nELEM = self.nELEM = nelx*nely
        nNODE = self.nNODE = (1+nelx) * (1+nely)
        nDOF = nNODE * 2

        (bpts_xy, areafraction, segLength) = lsm_solver.discretise()
        nBpts = self.nBpts = bpts_xy.shape[0]

        # 0. boundary condition setup (via Lagrange multiplier)
        (nodes, elem, elem_dof) = fea_solver.get_mesh()
        length_x = max(nodes[:, 0])
        length_y = max(nodes[:, 1])

        # SIMP_1. get boundary points
        comp_ = IndepVarComp()
        comp_.add_output('rhs', val = force)
        comp_.add_output('Vn', val = 0.0, shape=nBpts)
        self.add_subsystem('inputs_comp', comp_)
        # self.connect('inputs_comp.bpts', 'area_comp.points')
        self.connect('inputs_comp.Vn', 'area_comp.Vn')
        self.connect('inputs_comp.rhs', 'states_comp.rhs')

        # SIMP_2. boundary-to-area
        # comp_ = DiscretizeComp(lsm_solver=lsm_solver, nelx=nelx, nely=nely,
        #                             nBpts=nBpts, perturb=0.2)
        comp_ = VnPerturbComp(lsm_solver=lsm_solver, nelx=nelx, nely=nely,
                                nBpts=nBpts, perturb=0.2)
        self.add_subsystem('area_comp', comp_)
        self.connect('area_comp.density', 'states_comp.multipliers')
        self.connect('area_comp.density', 'weight_comp.x')
        self.connect('area_comp.density', 'VMstress_comp.density')

        # SIMP_3. area-to-state
        comp_ = SIMP_StatesComp(fem_solver=fea_solver, num_nodes_x=nelx+1, num_nodes_y=nely+1, isSIMP=True)
        self.add_subsystem('states_comp', comp_)
        self.connect('states_comp.states', 'disp_comp.states')

        # SIMP_4. extract out lagrange multipliers
        comp_ = SIMP_DispComp(num_nodes_x = nelx+1, num_nodes_y = nely+1)
        self.add_subsystem('disp_comp', comp_)
        self.connect('disp_comp.disp', 'VMstress_comp.disp')

        # SIMP_5. VM stress
        comp_ = VMStressComp(fea_solver=fea_solver, nelx=nelx, nely=nely, length_x=length_x, length_y=length_y, order=1.0, E=E, nu=nu)
        self.add_subsystem('VMstress_comp', comp_)
        self.connect('VMstress_comp.vmStress', 'pVM_comp.x')

        # SIMP_5.1. VM^p stress
        comp_ = pVmComp(pval=pval, nelx=nelx, nely=nely)
        self.add_subsystem('pVM_comp', comp_)
        self.connect('pVM_comp.xp', 'Integ_pVm_comp.x')

        # SIMP_5.2. int_omega VM^p
        comp_ = BodyIntegComp(nelx=nelx, nely=nely, length_x=length_x, length_y=length_y) # partial verified
        self.add_subsystem('Integ_pVm_comp', comp_)
        self.connect('Integ_pVm_comp.y', 'pnorm_comp.x')

        # SIMP_5.3. pnorm
        comp_ = pnormComp(nelx=nelx, nely=nely, pval=pval) # partial verified
        self.add_subsystem('pnorm_comp', comp_)
        self.add_objective('pnorm_comp.pnorm')

        # SIMP_6. total area
        comp_ = SIMP_WeightComp(num=nELEM)
        self.add_subsystem('weight_comp', comp_)
        self.add_constraint('weight_comp.weight', upper = 0.4*nelx*nely)

        # dummy
        self.add_design_var('inputs_comp.Vn') # without this, total derivative is not calculated

class HeatCouplingGroup(Group):
    def initialize(self):
        self.options.declare('fea_solver', types=py_FEA)
        self.options.declare('lsm_solver',types=py_LSM )
        self.options.declare('nelx', types=int)
        self.options.declare('nely', types=int)
        self.options.declare('force_e', types= np.ndarray) # f_el
        self.options.declare('force_t', types= np.ndarray) # thermal load (heat gen.)
        self.options.declare('movelimit', types= float)
        self.options.declare('K_cond', types= float)
        self.options.declare('BCid_e', types= np.ndarray) # elastic BC
        self.options.declare('BCid_t', types= np.ndarray) # heat sink
        self.options.declare('E', types= float)
        self.options.declare('nu', types= float)
        self.options.declare('alpha', types= float)
        self.options.declare('w', types=float) # weight between heat/el compliances


    def setup(self):
        self.lsm_solver = lsm_solver = self.options['lsm_solver']
        self.fea_solver = fea_solver = self.options['fea_solver']
        self.force_e = force_e = self.options['force_e']
        self.force_t = force_t = self.options['force_t']
        self.nelx = nelx = self.options['nelx']
        self.nely = nely = self.options['nely']
        self.movelimit = movelimit = self.options['movelimit']
        self.K_cond = K_cond = self.options['K_cond']
        self.BCid_e = BCid_e = self.options['BCid_e']
        self.BCid_t = BCid_t = self.options['BCid_t']
        self.E = E = self.options['E']
        self.nu = nu = self.options['nu']
        self.alpha = alpha = self.options['alpha']
        self.w = w = self.options['w']

        nELEM = self.nELEM = nelx * nely
        nNODE = self.nNODE = (nelx+1) * (nely+1)
        nDOF_e = self.nNODE * 2
        nDOF_t = self.nNODE
        nDOF_e_wLambda = self.nNODE * 2 + len(BCid_e)
        nDOF_t_wLambda = self.nNODE + len(BCid_t)

        phi = lsm_solver.get_phi()
        (bpts_xy, areafraction, segLength) = lsm_solver.discretise()
        nBpts = self.nBpts = bpts_xy.shape[0]

        (nodes, elem, elem_dof) = fea_solver.get_mesh()
        length_x = max(nodes[:,0])
        length_y = max(nodes[:,1])

        # 1. get boundary points
        comp_ = IndepVarComp()
        comp_.add_output('rhs_e', val = force_e) # mechanical load s(fixed load)
        comp_.add_output('rhs_t', val = force_t) # thermal load (heat creation)
        comp_.add_output('Vn', val = 0.0, shape=nBpts)
        self.add_subsystem('inputs_comp', comp_)
        # self.connect('inputs_comp.bpts', 'area_comp.points')
        self.connect('inputs_comp.Vn', 'area_comp.Vn')
        self.connect('inputs_comp.rhs_e', 'coupledload_comp.f_m')
        self.connect('inputs_comp.rhs_t', 'conductivity_comp.rhs')
        #self.connect('inputs_comp.rhs_e', 'GF_e_comp.states')
        self.connect('inputs_comp.rhs_t', 'GF_t_comp.states')

        # 2. boundary-to-area
        # comp_ = DiscretizeComp(lsm_solver=lsm_solver, nelx=nelx, nely=nely,
        #                             nBpts=nBpts, perturb=0.2)
        comp_ = VnPerturbComp(lsm_solver=lsm_solver, nelx=nelx, nely=nely,
                                nBpts=nBpts, perturb=0.2)
        self.add_subsystem('area_comp', comp_)
        self.connect('area_comp.density', 'conductivity_comp.multipliers')
        self.connect('area_comp.density', 'elasticity_comp.multipliers')
        self.connect('area_comp.density', 'coupledload_comp.multipliers')
        self.connect('area_comp.density', 'weight_comp.x')

        # COND_3.0. area-to-state (conduction)
        comp_ = ConductComp(k_cond = K_cond, BCid = BCid_t, nelx=nelx, nely=nely, length_x=length_x, length_y=length_y, ELEM = elem)
        self.add_subsystem('conductivity_comp', comp_)
        self.connect('conductivity_comp.states', 'temp_comp.states')

        # SIMP_4.0 extract out lagrange multipliers
        comp_ = COND_DispComp(num_nodes_x = nelx+1, num_nodes_y = nely+1, nBCid=len(BCid_t))
        self.add_subsystem('temp_comp', comp_)
        self.connect('temp_comp.disp', 'compliance_t_comp.disp')
        self.connect('temp_comp.disp', 'coupledload_comp.temperatures')

        comp_ = COND_DispComp(num_nodes_x = nelx+1, num_nodes_y = nely+1, nBCid=len(BCid_t))
        self.add_subsystem('GF_t_comp', comp_)
        self.connect('GF_t_comp.disp', 'compliance_t_comp.forces')

        # COUPLE_3.2. coupled therm.
        comp_ = ThermCoupleLoadComp(alpha=alpha, nelx=nelx, nely=nely, length_x=length_x, length_y=length_y, ELEM=elem, DOF_m2do=elem_dof, E=E, nu=nu, BCid=BCid_e)
        self.add_subsystem('coupledload_comp', comp_)
        self.connect('coupledload_comp.f_tot', 'elasticity_comp.rhs')
        self.connect('coupledload_comp.f_tot', 'GF_et_comp.states')

        # SIMP_3.1. area-to-state (elasticity)
        comp_ = SIMP_StatesComp(fem_solver=fea_solver, num_nodes_x=nelx+1, num_nodes_y=nely+1, isSIMP=True)
        self.add_subsystem('elasticity_comp', comp_)
        self.connect('elasticity_comp.states', 'disp_comp.states')

        # SIMP_4.1. extract out lagrange multipliers
        comp_ = SIMP_DispComp(num_nodes_x = nelx+1, num_nodes_y = nely+1, nBCid = len(BCid_e))
        self.add_subsystem('disp_comp', comp_)
        self.connect('disp_comp.disp', 'compliance_et_comp.disp')

        comp_ = SIMP_DispComp(num_nodes_x = nelx+1, num_nodes_y = nely+1, nBCid = len(BCid_e))
        self.add_subsystem('GF_et_comp', comp_)
        self.connect('GF_et_comp.disp', 'compliance_et_comp.forces')

        # COND_5.0 compliance
        comp_ = COND_ComplianceComp(num_nodes_x = nelx+1, num_nodes_y = nely+1)
        self.add_subsystem('compliance_t_comp', comp_)
        self.connect('compliance_t_comp.compliance', 'objective_comp.x1')

        # SIMP_5.1 compliance
        comp_ = SIMP_ComplianceComp(num_nodes_x = nelx+1, num_nodes_y = nely+1)
        self.add_subsystem('compliance_et_comp', comp_)
        self.connect('compliance_et_comp.compliance', 'objective_comp.x2')
        # self.add_objective('compliance_et_comp.compliance')

        # 6. compliance
        comp_ = SumComp(w=w)
        self.add_subsystem('objective_comp', comp_)
        self.add_objective('objective_comp.y') 

        # 6. total area
        comp_ = SIMP_WeightComp(num=nELEM)
        self.add_subsystem('weight_comp', comp_)
        self.add_constraint('weight_comp.weight', upper = 0.4*nelx*nely)

        # dummy
        self.add_design_var('inputs_comp.Vn') # without this, total derivative is not calculated

class ConductionGroup(Group):
    def initialize(self):
        self.options.declare('fea_solver', types=py_FEA)
        self.options.declare('lsm_solver',types=py_LSM )
        self.options.declare('nelx', types=int)
        self.options.declare('nely', types=int)
        self.options.declare('force', types= np.ndarray)
        self.options.declare('movelimit', types= float)
        self.options.declare('K_cond', types= float)
        self.options.declare('BCid', types= np.ndarray)

    def setup(self):
        self.lsm_solver = lsm_solver = self.options['lsm_solver']
        self.fea_solver = fea_solver = self.options['fea_solver']
        self.force = force = self.options['force']
        self.nelx = nelx = self.options['nelx']
        self.nely = nely = self.options['nely']
        self.movelimit = movelimit = self.options['movelimit']
        self.K_cond = K_cond = self.options['K_cond']
        self.BCid = BCid = self.options['BCid']

        phi = lsm_solver.get_phi()
        nELEM = self.nELEM = nelx*nely
        nNODE = self.nNODE = (1+nelx) * (1+nely)
        nDOF = nNODE

        (bpts_xy, areafraction, segLength) = lsm_solver.discretise()
        nBpts = self.nBpts = bpts_xy.shape[0]

        (nodes, elem, elem_dof) = fea_solver.get_mesh()
        length_x = max(nodes[:, 0])
        length_y = max(nodes[:, 1])

        # SIMP_1. get boundary points
        comp_ = IndepVarComp()
        comp_.add_output('rhs', val = force)
        comp_.add_output('Vn', val = 0.0, shape=nBpts)
        self.add_subsystem('inputs_comp', comp_)
        # self.connect('inputs_comp.bpts', 'area_comp.points')
        self.connect('inputs_comp.Vn', 'area_comp.Vn')
        self.connect('inputs_comp.rhs', 'states_comp.rhs')
        self.connect('inputs_comp.rhs', 'GF_comp.states')

        # SIMP_2. boundary-to-area
        # comp_ = DiscretizeComp(lsm_solver=lsm_solver, nelx=nelx, nely=nely,
        #                             nBpts=nBpts, perturb=0.2)
        comp_ = VnPerturbComp(lsm_solver=lsm_solver, nelx=nelx, nely=nely,
                                nBpts=nBpts, perturb=0.2)
        self.add_subsystem('area_comp', comp_)
        self.connect('area_comp.density', 'states_comp.multipliers')
        self.connect('area_comp.density', 'weight_comp.x')

        # SIMP_3. area-to-state
        comp_ = ConductComp(k_cond = K_cond, BCid = BCid, nelx=nelx, nely=nely, length_x=length_x, length_y=length_y, ELEM = elem)
        self.add_subsystem('states_comp', comp_)
        self.connect('states_comp.states', 'temp_comp.states')

        # SIMP_4. extract out lagrange multipliers
        comp_ = COND_DispComp(num_nodes_x = nelx+1, num_nodes_y = nely+1, nBCid=len(BCid))
        self.add_subsystem('temp_comp', comp_)
        self.connect('temp_comp.disp', 'compliance_comp.disp')

        comp_ = COND_DispComp(num_nodes_x = nelx+1, num_nodes_y = nely+1, nBCid=len(BCid))
        self.add_subsystem('GF_comp', comp_)
        self.connect('GF_comp.disp', 'compliance_comp.forces')

        # SIMP_5. compliance
        comp_ = COND_ComplianceComp(num_nodes_x = nelx+1, num_nodes_y = nely+1)
        self.add_subsystem('compliance_comp', comp_)
        self.add_objective('compliance_comp.compliance')

        # SIMP_6. total area
        comp_ = SIMP_WeightComp(num=nELEM)
        self.add_subsystem('weight_comp', comp_)
        self.add_constraint('weight_comp.weight', upper = 0.4*nelx*nely)

        self.add_design_var('inputs_comp.Vn') # without this, total derivative is not calculated


class ComplianceGroup(Group):

    def initialize(self):
        self.options.declare('fea_solver', types=py_FEA)
        self.options.declare('lsm_solver',types=py_LSM )
        self.options.declare('nelx', types=int)
        self.options.declare('nely', types=int)
        self.options.declare('force', types= np.ndarray)
        self.options.declare('movelimit', types= float)
        self.options.declare('BCid', types = np.ndarray)

    def setup(self):
        self.lsm_solver = lsm_solver = self.options['lsm_solver']
        self.fea_solver = fea_solver = self.options['fea_solver']
        self.force = force = self.options['force']
        self.nelx = nelx = self.options['nelx']
        self.nely = nely = self.options['nely']
        self.movelimit = movelimit = self.options['movelimit']
        self.BCid = BCid = self.options['BCid']
        self.nBCid = nBCid = BCid.shape[0]

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
        comp_.add_output('Vn', val = 0.0, shape=nBpts)
        self.add_subsystem('inputs_comp', comp_)
        # self.connect('inputs_comp.bpts', 'area_comp.points')
        self.connect('inputs_comp.Vn', 'area_comp.Vn')
        self.connect('inputs_comp.rhs', 'states_comp.rhs')
        self.connect('inputs_comp.rhs', 'GF_comp.states')

        # SIMP_2. boundary-to-area
        # comp_ = DiscretizeComp(lsm_solver=lsm_solver, nelx=nelx, nely=nely,
        #                             nBpts=nBpts, perturb=0.2)
        comp_ = VnPerturbComp(lsm_solver=lsm_solver, nelx=nelx, nely=nely,
                                nBpts=nBpts, perturb=0.2)
        self.add_subsystem('area_comp', comp_)
        self.connect('area_comp.density', 'states_comp.multipliers')
        self.connect('area_comp.density', 'weight_comp.x')

        # SIMP_3. area-to-state
        comp_ = SIMP_StatesComp(fem_solver=fea_solver, num_nodes_x=nelx+1, num_nodes_y=nely+1, isSIMP=True)
        self.add_subsystem('states_comp', comp_)
        self.connect('states_comp.states', 'disp_comp.states')

        # SIMP_4. extract out lagrange multipliers
        comp_ = SIMP_DispComp(num_nodes_x = nelx+1, num_nodes_y = nely+1, nBCid = self.nBCid)

        self.add_subsystem('disp_comp', comp_)
        self.connect('disp_comp.disp', 'compliance_comp.disp')

        comp_ = SIMP_DispComp(num_nodes_x = nelx+1, num_nodes_y = nely+1, nBCid = self.nBCid)

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
        self.add_constraint('weight_comp.weight', upper = 0.4*nelx*nely)


        # fixme: solve_linear() is called so many times
        # dummy
        self.add_design_var('inputs_comp.Vn') # without this, total derivative is not calculated


        # self.options['assembled_jac_type'] = 'csc'
        # self.linear_solver = DirectSolver(assemble_jac=True)
