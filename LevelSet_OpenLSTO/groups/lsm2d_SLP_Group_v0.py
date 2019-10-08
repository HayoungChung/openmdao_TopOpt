# this is deprecated version: constrain computation error
import numpy as np 

from openmdao.api import Group, IndepVarComp
from myExplicitComponents import AreaConstComp, ScaleComp, DispComp
from lsm_classes import PyLSMSolver

class LSM2D_slpGroup(Group):
    def initialize(self):
        self.options.declare('lsm_solver', types=PyLSMSolver, required=True)
        self.options.declare('bpts_xy', types=np.ndarray, required=True)
        self.options.declare('segmentLength', types=np.ndarray, required=True)
        # self.options.declare('area_fraction', types=np.ndarray, required=True)
        # self.options.declare('fixedGpts_xy', types=np.ndarray, required=True)
        # self.options.declare('fixedGpts_sens', types=np.ndarray, required=True)
        self.options.declare('radius', types=(int, float), required=True)
        # self.options.declare('movelimit', types=float, required=True)
        self.options.declare('bpts_sens', types=np.ndarray, required=True)
        self.options.declare('ub', types=(list,np.ndarray), required=True)
        self.options.declare('lb', types=(list,np.ndarray), required=True)

    def setup(self):
        lsm_solver = self.options['lsm_solver']
        bpts_xy = self.options['bpts_xy']
        segmentLength = self.options['segmentLength']
        # area_fraction = self.options['area_fraction']
        # fixedGpts_xy = self.options['fixedGpts_xy']
        # fixedGpts_sens = self.options['fixedGpts_sens']
        radius = self.options['radius']
        # movelimit = self.options['movelimit']
        bpts_sens = self.options['bpts_sens']
        upperbound = self.options['ub']
        lowerbound = self.options['lb']


        num_dvs = 2 # number of lambdas
        num_bpts = bpts_xy.shape[0]
        
        # inputs (IndepVarComp: component)
        comp = IndepVarComp()
        comp.add_output('lambdas', val = 0.0, shape = num_dvs)
        # comp.add_output('fixedGpts_sens', val = fixedGpts_sens)
        comp.add_output('bpts_sens', val = bpts_sens)
        self.add_subsystem('inputs_comp', comp)
        # self.connect('inputs_comp.fixedGpts_sens', 'compl_sens_comp.fixedGpts_sens')
        self.connect('inputs_comp.lambdas', 'constraint_comp.lambdas')
        self.connect('inputs_comp.lambdas', 'displacement_comp.lambdas')
        self.connect('inputs_comp.bpts_sens', 'compliance_scale_comp.x')
        
        # # compliance sensitivity at each boundary points
        # comp = ComplSensComp(lsm_solver = lsm_solver, 
        #             bpts_xy = bpts_xy, fixedGpts_xy = fixedGpts_xy, 
        #             radius = radius, area_fraction = area_fraction,
        #             movelimit = movelimit,
        #             tmpFix_bpts_sens = tmpFix_bpts_sens,)
        # self.add_subsystem('compl_sens_comp', comp)
        # self.connect('compl_sens_comp.bpts_sens', 'compliance_scale_comp.x')
                
        # self.add_design_var('inputs_comp.lambdas', 
        #     lower = np.array(['compl_sens_comp.lowerbound'[0], 'compl_sens_comp.lowerbound'[1]]), 
        #     upper = np.array(['compl_sens_comp.upperbound'[0], 'compl_sens_comp.upperbound'[1]]))

        self.add_design_var('inputs_comp.lambdas', 
            lower = np.array([lowerbound[0], lowerbound[1]]), 
            upper = np.array([upperbound[0], upperbound[1]]))

        
        # constraint setup
        comp = AreaConstComp(lsm_solver = lsm_solver, num_bpts = num_bpts, num_dvs = num_dvs)
        comp.add_constraint('G_cons')
        self.add_subsystem('constraint_comp', comp)
        self.connect('constraint_comp.bpts_sens', 'constraint_scale_comp.x')

        # normalize the (compliance sensitivity)
        comp = ScaleComp(num_bpts = num_bpts) #, num_dvs = num_dvs)
        self.add_subsystem('compliance_scale_comp', comp)
        src_indices_tmp = np.zeros((num_bpts,2),dtype=int)
        src_indices_tmp[:,0] = np.arange(num_bpts)
        self.connect('compliance_scale_comp.y', 'displacement_comp.sens_c')#, src_indices=src_indices_tmp)

        # normalize the (area sensitivity)
        comp = ScaleComp(num_bpts = num_bpts)#, num_dvs = num_dvs)
        self.add_subsystem('constraint_scale_comp', comp)
        src_indices_tmp = np.zeros((num_bpts,2),dtype=int)
        src_indices_tmp[:,1] = np.arange(num_bpts)
        self.connect('constraint_scale_comp.y', 'displacement_comp.sens_a')#,src_indices=src_indices_tmp)

        # displacement (z) calculation
        comp = DispComp(lsm_solver = lsm_solver, num_bpts = num_bpts, num_dvs = num_dvs, segmentLength = segmentLength) #lsm_module = lsm_module, lsm_mesh = lsm_mesh)
        self.add_subsystem('displacement_comp', comp)
        self.add_objective('displacement_comp.F_obj')


