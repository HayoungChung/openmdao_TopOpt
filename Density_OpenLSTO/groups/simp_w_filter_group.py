# this is a elem-wise simp
import numpy as np

from openmdao.api import Group, IndepVarComp

from components.penalization_comp import PenalizationComp
from components.heaviside_comp import HeavisideComp
from components.states_comp import StatesComp
from components.disp_comp import DispComp
from components.compliance_comp import ComplianceComp
from components.weight_comp import WeightComp
from components.objective_comp import ObjectiveComp

from pyBind import py_FEA

from utils.rbf import get_rbf_mtx
from utils.bspline import get_bspline_mtx

from components.densityfilter_comp import DensityFilterComp


class SimpGroup(Group):

    def initialize(self):
        self.options.declare('fem_solver', types=py_FEA, )#)#required=True)
        self.options.declare('force', types=np.ndarray, )#)#required=True)
        self.options.declare('num_elem_x', types=int, )#)#required=True)
        self.options.declare('num_elem_y', types=int, )#)#required=True)
        self.options.declare('penal', types=(int, float), )#)#required=True)
        self.options.declare(
            'volume_fraction', types=(int, float), )#)#required=True)

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
        comp = IndepVarComp()
        comp.add_output('rhs', val=rhs)
        comp.add_output('dvs', val=0.4, shape=nELEM)
        comp.add_design_var('dvs', lower=0.01, upper=1.0)
        # comp.add_design_var('x', lower=-4, upper=4) // param
        self.add_subsystem('inputs_comp', comp)
        self.connect('inputs_comp.dvs', 'filter_comp.dvs')
        self.connect('inputs_comp.rhs', 'states_comp.rhs')

        # density filter
        comp = DensityFilterComp(length_x=length_x, length_y=length_y, 
                                num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y,
                                 num_dvs=nELEM, radius=length_x / (float(num_nodes_x) - 1) * 2)
        self.add_subsystem('filter_comp', comp)
        self.connect('filter_comp.dvs_bar', 'penalization_comp.x')
        self.connect('filter_comp.dvs_bar', 'weight_comp.x')

        # penalization
        comp = PenalizationComp(num=nELEM, p=p)
        self.add_subsystem('penalization_comp', comp)
        self.connect('penalization_comp.y', 'states_comp.multipliers')

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
        comp.add_constraint('weight', upper=volume_fraction)
        self.add_subsystem('weight_comp', comp)

        # objective
        comp = ObjectiveComp(w=0.0)
        comp.add_objective('objective')
        self.add_subsystem('objective_comp', comp)
        self.connect('compliance_comp.compliance', 'objective_comp.compliance')
        self.connect('weight_comp.weight', 'objective_comp.weight')


