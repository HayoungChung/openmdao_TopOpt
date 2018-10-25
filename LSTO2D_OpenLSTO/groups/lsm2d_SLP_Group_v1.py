import numpy as np 

from openmdao.api import Group, IndepVarComp
from myExplicitComponents import Callback_objF, Callback_conF #DistanceComp, ConstraintComp, ObjectiveComp #,AreaConstComp, ScaleComp, DispComp
from lsm_classes import PyLSMSolver

class LSM2D_slpGroup(Group):
    def initialize(self):
        self.metadata.declare('lsm_solver', type_=PyLSMSolver, required=True)
        self.metadata.declare('bpts_xy', type_=np.ndarray, required=True)
        self.metadata.declare('ub', type_=(list,np.ndarray), required=True)
        self.metadata.declare('lb', type_=(list,np.ndarray), required=True)

    def setup(self):
        lsm_solver = self.metadata['lsm_solver']
        bpts_xy = self.metadata['bpts_xy']
        upperbound = self.metadata['ub']
        lowerbound = self.metadata['lb']

        num_dvs = 2 # number of lambdas
        num_bpts = bpts_xy.shape[0]
        
        # inputs (IndepVarComp: component)
        comp = IndepVarComp()
        comp.add_output('lambdas', val = 0.0, shape = num_dvs)
        self.add_subsystem('inputs_comp', comp)        
        self.connect('inputs_comp.lambdas', 'objective_comp.lambdas')
        self.connect('inputs_comp.lambdas', 'constraint_comp.lambdas')
        
        self.add_design_var('inputs_comp.lambdas', 
            lower = np.array([lowerbound[0], lowerbound[1]]), 
            upper = np.array([upperbound[0], upperbound[1]]))
        
        # constraint setup
        comp = Callback_conF(lsm_solver = lsm_solver)
        comp.add_constraint('constraint')
        self.add_subsystem('constraint_comp', comp)
        
        # objective setup
        comp = Callback_objF(lsm_solver = lsm_solver)
        comp.add_objective('objective')
        self.add_subsystem('objective_comp', comp)
                

        
        
