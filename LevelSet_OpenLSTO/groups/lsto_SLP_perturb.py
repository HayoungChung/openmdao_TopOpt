# This is a group that optimizes V_n for given phi...
# this leads to a larger code, where H-J is solved 

import numpy as np

from openmdao.api import Group, IndepVarComp

from ../Density_OpenLSTO/components.states_comp import StatesComp
from ../Density_OpenLSTO/components.disp_comp import DispComp
from ../Density_OpenLSTO/components.compliance_comp import ComplianceComp
from ../Density_OpenLSTO/components.weight_comp import WeightComp
from ../Density_OpenLSTO/components.objective_comp import ObjectiveComp
from components_new.IntegralComp import *
from components_new.DisplacementComp import *
from components_new.ObjectiveComp import *
from components_new.ConstraintComp import *
from components_new.ScalingComp import *

from pyBind import py_FEA
from py_lsmBind import py_LSM

class PerturbGroup(Gropup):
    def initialize(self):
        pass   

    def setup(self):
        pass

    def 

class StateGroup(Group):

    def initialize(self):
        self.options.declare('fem_solver', types=py_FEA, )
        self.options.declare('force', types=np.ndarray, )
        self.options.declare('num_elem_x', types=int, )
        self.options.declare('num_elem_y', types=int, )

    def setup(self):
        fem_solver = self.options['fem_solver']
        force = self.options['force']
        num_elem_x = self.options['num_elem_x']
        num_elem_y = self.options['num_elem_y']
        p = self.options['penal']
        volume_fraction = self.options['volume_fraction']
        (nodes, elem, elem_dof) = fem_solver.get_mesh()

        (length_x, length_y) = (np.max(nodes[:, 0], 0), np.max(nodes[:, 1], 0))
        (num_nodes_x, num_nodes_y) = (num_elem_x + 1, num_elem_y + 1)

        nNODE = num_nodes_x * num_nodes_y
        nELEM = (num_nodes_x - 1) * (num_nodes_y - 1)
        nDOF = nNODE * 2

        rhs = force

        # inputs
        # comp = IndepVarComp()
        # comp.add_output('rhs', val=rhs)
        # comp.add_output('dvs', val=0.8, shape=nELEM)
        # comp.add_design_var('dvs', lower=0.01, upper=1.0)
        # # comp.add_design_var('x', lower=-4, upper=4) // param
        # self.add_subsystem('inputs_comp', comp)
        # self.connect('inputs_comp.dvs', 'penalization_comp.x')
        # self.connect('inputs_comp.dvs', 'weight_comp.x')
        # self.connect('inputs_comp.rhs', 'states_comp.rhs')

        comp = 

        # states 
        comp = StatesComp(fem_solver=fem_solver, 
                          num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y,
                          isSIMP = True)
        self.add_subsystem('states_comp', comp)
        self.connect('states_comp.states', 'disp_comp.states')

        # num_states to num_dofs
        comp = DispComp(num_nodes_x=num_nodes_x,num_nodes_y=num_nodes_y)
        self.add_subsystem('disp_comp',comp)
        self.connect('disp_comp.disp', 'compliance_comp.disp')

        comp = DispComp(num_nodes_x=num_nodes_x,num_nodes_y=num_nodes_y)
        self.add_subsystem('GF_comp',comp)
        self.connect('inputs_comp.rhs', 'GF_comp.states')
        self.connect('GF_comp.disp', 'compliance_comp.forces')

        # compliance
        comp = ComplianceComp(num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y)
        self.add_subsystem('compliance_comp', comp)
        # self.connect('inputs_comp.rhs', 'compliance_comp.forces')

        # weight
        comp = WeightComp(num=nELEM)
        # comp.add_constraint('weight', upper=volume_fraction)
        self.add_subsystem('weight_comp', comp)

