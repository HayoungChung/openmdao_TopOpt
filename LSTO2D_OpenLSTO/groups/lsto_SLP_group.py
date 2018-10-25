import numpy as np 

from openmdao.api import Group, IndepVarComp
from components.myExplicitComponents import DisplacementComp, ConstraintComp, ObjectiveComp #,AreaConstComp, ScaleComp, DispComp
# from lsm_classes import PyLSMSolver
from py_lsmBind import py_LSM


class LSM2D_slpGroup(Group):
    def initialize(self):
        self.metadata.declare('lsm_solver', type_=py_LSM, required=True)
        self.metadata.declare('bpts_xy', type_=np.ndarray, required=True)
        self.metadata.declare('bpts_sens', type_=np.ndarray, required=True)
        self.metadata.declare('ActiveList', type_=np.ndarray, required=True)
        self.metadata.declare('length_segs', type_=np.ndarray, required=True)
        self.metadata.declare('ub', type_=(list,np.ndarray), required=True)
        self.metadata.declare('lb', type_=(list,np.ndarray), required=True)

    def setup(self):
        lsm_solver = self.metadata['lsm_solver']
        bpts_xy = self.metadata['bpts_xy']
        bpts_sens = self.metadata['bpts_sens']
        activeList = self.metadata['ActiveList']
        length_segs = self.metadata['length_segs']
        upperbound = self.metadata['ub']
        lowerbound = self.metadata['lb']

        num_dvs = 2 # number of lambdas
        num_bpts = bpts_xy.shape[0]
        Sf = bpts_sens[:,0]
        Sg = bpts_sens[:,1]
        
        # inputs (IndepVarComp: component)
        comp = IndepVarComp()
        comp.add_output('lambdas', val = 0.0, shape = num_dvs)
        comp.add_output('Sf', val = Sf)
        comp.add_output('Sg', val = Sg)
        comp.add_output('length', val = length_segs)

        self.add_subsystem('inputs_comp', comp)        
        self.add_design_var('inputs_comp.lambdas', 
            lower = np.array([lowerbound[0], lowerbound[1]]), 
            upper = np.array([upperbound[0], upperbound[1]]))
        self.connect('inputs_comp.lambdas', 'displacement_comp.lambdas')
        self.connect('inputs_comp.Sf', 'active_Sf_comp.x')
        self.connect('inputs_comp.Sg', 'active_Sg_comp.x')
        self.connect('inputs_comp.length', 'integration_f_comp.x')
        self.connect('inputs_comp.length', 'integration_g_comp.x')
        
        # active nodes only
        comp = ActiveComp(activeid = activeList, nBpts = num_bpts)
        self.add_subsystem('active_Sf_comp', comp)
        self.connect('active_Sf_comp.y', 'integration_f_comp.x')

        comp = ActiveComp(activeid = activeList, nBpts = num_bpts)
        self.add_subsystem('active_Sg_comp', comp)
        self.connect('active_Sg_comp.y', 'integration_g_comp.x')

        # integrations
        comp = IntegComp(nBpts = num_bpts)
        self.add_subsystem('active_Sf_comp', comp)
        self.connect('active_Sf_comp.y', 'integration_f_comp.x')

        comp = IntegComp(nBpts = num_bpts)
        self.add_subsystem('active_Sg_comp', comp)
        self.connect('active_Sg_comp.y', 'integration_g_comp.x')
        
        # displacements setup
        comp = DisplacementComp(lsm_solver = lsm_solver, num_bpts = num_bpts, 
            num_dvs = num_dvs)
        self.add_subsystem('displacement_comp', comp)
        self.connect('displacement_comp.displacements', 'objective_comp.displacements')
        self.connect('displacement_comp.displacements', 'constraint_comp.displacements')
        
        # objective setup
        comp = ObjectiveComp(lsm_solver = lsm_solver, num_bpts = num_bpts)
        comp.add_objective('objective')
        self.add_subsystem('objective_comp', comp)
                
        # constraint setup
        comp = ConstraintComp(lsm_solver = lsm_solver, num_bpts = num_bpts)
        comp.add_constraint('constraint', upper = 0.0 )
        self.add_subsystem('constraint_comp', comp)

        
        
