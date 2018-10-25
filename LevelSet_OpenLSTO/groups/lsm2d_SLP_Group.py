import numpy as np 

from openmdao.api import Group, IndepVarComp
from components.myExplicitComponents import DisplacementComp, ConstraintComp, ObjectiveComp #,AreaConstComp, ScaleComp, DispComp
from lsm_classes import PyLSMSolver

class LSM2D_slpGroup(Group):
    def initialize(self):
        self.options.declare('lsm_solver', types=PyLSMSolver, )#required=True)
        self.options.declare('bpts_xy', types=np.ndarray, )#required=True)
        self.options.declare('ub', types=(list,np.ndarray), )#required=True)
        self.options.declare('lb', types=(list,np.ndarray), )#required=True)

    def setup(self):
        lsm_solver = self.options['lsm_solver']
        bpts_xy = self.options['bpts_xy']
        upperbound = self.options['ub']
        lowerbound = self.options['lb']

        num_dvs = 2 # number of lambdas
        num_bpts = bpts_xy.shape[0]
        
        # inputs (IndepVarComp: component)
        comp = IndepVarComp()
        comp.add_output('lambdas', val = 0.0, shape = num_dvs)
        self.add_subsystem('inputs_comp', comp)        
        self.connect('inputs_comp.lambdas', 'displacement_comp.lambdas')
                
        self.add_design_var('inputs_comp.lambdas', 
            lower = np.array([lowerbound[0], lowerbound[1]]), 
            upper = np.array([upperbound[0], upperbound[1]]))
        
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

        
        
