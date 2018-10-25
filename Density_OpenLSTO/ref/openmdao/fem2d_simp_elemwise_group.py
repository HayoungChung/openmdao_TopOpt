# this is a elem-wise simp
import numpy as np

from openmdao.api import Group, IndepVarComp

from fem2d.openmdao.penalization_comp import PenalizationComp
from fem2d.openmdao.heaviside_comp import HeavisideComp
from fem2d.openmdao.states_comp import StatesComp
from fem2d.openmdao.disp_comp import DispComp
from fem2d.openmdao.compliance_comp import ComplianceComp
from fem2d.openmdao.weight_comp import WeightComp
from fem2d.openmdao.objective_comp import ObjectiveComp
from fem2d.fem2d import PyFEMSolver
from fem2d.utils.rbf import get_rbf_mtx
from fem2d.utils.bspline import get_bspline_mtx


class FEM2DSimpGroup(Group):

    def initialize(self):
        self.options.declare('fem_solver', types=PyFEMSolver, )#required=True)
        self.options.declare('length_x', types=(int, float), )#required=True)
        self.options.declare('length_y', types=(int, float), )#required=True)
        self.options.declare('num_nodes_x', types=int, )#required=True)
        self.options.declare('num_nodes_y', types=int, )#required=True)
        self.options.declare('forces', types=np.ndarray, )#required=True)
        self.options.declare('p', types=(int, float), )#required=True)
        self.options.declare('w', types=(int, float), )#required=True)
        self.options.declare('nodes', types=np.ndarray, )#required=True)
        self.options.declare('volume_fraction', types=(int, float), )#required=True)

    def setup(self):
        fem_solver = self.options['fem_solver']
        length_x = self.options['length_x']
        length_y = self.options['length_y']
        num_nodes_x = self.options['num_nodes_x']
        num_nodes_y = self.options['num_nodes_y']
        forces = self.options['forces']
        p = self.options['p']
        w = self.options['w']
        nodes = self.options['nodes']
        volume_fraction = self.options['volume_fraction']

        num = num_nodes_x * num_nodes_y
        num_el = (num_nodes_x - 1) * (num_nodes_y - 1)

        state_size = 2 * num_nodes_x * num_nodes_y + 2 * num_nodes_y
        disp_size = 2 * num_nodes_x * num_nodes_y

        rhs = np.zeros(state_size)
        rhs[:disp_size] = forces

        # inputs
        comp = IndepVarComp()
        comp.add_output('rhs', val=rhs)
        comp.add_output('forces', val=forces)

        comp.add_output('dvs', val=0.5, shape=num_el)
        comp.add_design_var('dvs', lower=0.01, upper=1.0)
        # comp.add_design_var('x', lower=-4, upper=4)
        self.add_subsystem('inputs_comp', comp)
        self.connect('inputs_comp.dvs', 'penalization_comp.x')
        self.connect('inputs_comp.dvs', 'weight_comp.x')

        # penalization
        comp = PenalizationComp(num=num_el, p=p)
        self.add_subsystem('penalization_comp', comp)

        self.connect('penalization_comp.y', 'states_comp.multipliers')

        # states
        comp = StatesComp(fem_solver=fem_solver, num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y,
            nodes=nodes, isNodal=False)
        self.add_subsystem('states_comp', comp)
        self.connect('inputs_comp.rhs', 'states_comp.rhs')

        # disp
        comp = DispComp(num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y)
        self.add_subsystem('disp_comp', comp)
        self.connect('states_comp.states', 'disp_comp.states')

        # compliance
        comp = ComplianceComp(num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y)
        self.add_subsystem('compliance_comp', comp)
        self.connect('disp_comp.disp', 'compliance_comp.disp')
        self.connect('inputs_comp.forces', 'compliance_comp.forces')

        # weight
        comp = WeightComp(num=num_el)
        comp.add_constraint('weight', upper=volume_fraction)
        self.add_subsystem('weight_comp', comp)

        # objective
        comp = ObjectiveComp(w=w)
        comp.add_objective('objective')
        self.add_subsystem('objective_comp', comp)
        self.connect('compliance_comp.compliance', 'objective_comp.compliance')
        self.connect('weight_comp.weight', 'objective_comp.weight')
