import numpy as np 

from openmdao.api import Group, IndepVarComp
# from components.myExplicitComponents import DisplacementComp, ConstraintComp, ObjectiveComp #,AreaConstComp, ScaleComp, DispComp
from components_new.IntegralComp import *
from components_new.DisplacementComp import *
from components_new.ObjectiveComp import *
from components_new.ConstraintComp import *
from components_new.ScalingComp import *

class LSM2D_slpGroup(Group):
    def initialize(self):
        self.metadata.declare('lsm_solver', type_=py_LSM, )#required=True)
        self.metadata.declare('num_bpts', type_=(int,float), )#required=True)
        self.metadata.declare('ub', type_=(list,np.ndarray), )#required=True)
        self.metadata.declare('lb', type_=(list,np.ndarray), )#required=True)

        self.metadata.declare('Sf', type_=np.ndarray, )#required=True)
        self.metadata.declare('Sg', type_=np.ndarray, )#required=True)
        self.metadata.declare('constraintDistance', type_=float, )#required=True)


    def setup(self):
        lsm_solver = self.metadata['lsm_solver']
        num_bpts = self.metadata['num_bpts']
        upperbound = self.metadata['ub']
        lowerbound = self.metadata['lb']

        Sf = self.metadata['Sf']
        Sg = self.metadata['Sg']
        constraintDistance = self.metadata['constraintDistance']

        num_dvs = 2 # number of lambdas
        
        # inputs (IndepVarComp: component)
        comp = IndepVarComp()
        comp.add_output('lambdas', val = 0.0, shape = num_dvs)
        comp.add_output('Sf', val=Sf)
        comp.add_output('Sg', val=Sg)
        comp.add_output('constraintDistance', val=constraintDistance)

        self.add_subsystem('inputs_comp', comp)        
        self.add_design_var('inputs_comp.lambdas', 
            lower = np.array([lowerbound[0], lowerbound[1]]), 
            upper = np.array([upperbound[0], upperbound[1]]))

        # scalings setup # verified (10/24)
        comp = ScalingComp(nBpts= num_bpts, lsm_solver=lsm_solver)
        self.add_subsystem('scaling_f_comp',comp)
        self.connect('inputs_comp.Sf', 'scaling_f_comp.x') 
        
        comp = ScalingComp(nBpts= num_bpts, lsm_solver=lsm_solver)
        self.add_subsystem('scaling_g_comp',comp)
        self.connect('inputs_comp.Sg', 'scaling_g_comp.x')

        # displacements setup
        comp = DisplacementComp(lsm_solver = lsm_solver, nBpts = num_bpts, ndvs = num_dvs)
        self.add_subsystem('displacement_comp', comp)
        
        self.connect('inputs_comp.lambdas', 'displacement_comp.lambdas')
        self.connect('inputs_comp.Sf', 'displacement_comp.Sf')
        self.connect('inputs_comp.Sg', 'displacement_comp.Sg')

        self.connect('scaling_f_comp.y', 'displacement_comp.Scale_f')
        self.connect('scaling_g_comp.y', 'displacement_comp.Scale_g')

        # integration setup
        comp = IntegralComp(lsm_solver=lsm_solver, nBpts=num_bpts)
        self.add_subsystem('integral_f_comp', comp)
        self.connect('inputs_comp.Sf', 'integral_f_comp.x')

        comp = IntegralComp(lsm_solver=lsm_solver, nBpts=num_bpts)
        self.add_subsystem('integral_g_comp', comp)
        self.connect('inputs_comp.Sg', 'integral_g_comp.x')
        
        # objective setup
        comp = ObjectiveComp(lsm_solver = lsm_solver, nBpts = num_bpts)
        comp.add_objective('delF')
        self.add_subsystem('objective_comp', comp)
        
        self.connect('displacement_comp.displacements', 'objective_comp.displacements')
        self.connect('integral_f_comp.y', 'objective_comp.Cf')
        self.connect('scaling_f_comp.y', 'objective_comp.scale_f')


        # constraint setup
        comp = ConstraintComp(lsm_solver = lsm_solver, nBpts = num_bpts)
        comp.add_constraint('delG', upper = 0.0 )
        self.add_subsystem('constraint_comp', comp)

        self.connect('displacement_comp.displacements', 'constraint_comp.displacements')
        self.connect('integral_g_comp.y', 'constraint_comp.Cg')
        self.connect('scaling_g_comp.y', 'constraint_comp.scale_g')
        self.connect('inputs_comp.constraintDistance', 'constraint_comp.constraintDistance')

        
