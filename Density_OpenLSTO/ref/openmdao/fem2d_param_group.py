import numpy as np

from openmdao.api import Group, IndepVarComp

from fem2d.openmdao.parametrization_comp import ParametrizationComp
from fem2d.openmdao.averaging_comp import AveragingComp
from fem2d.openmdao.heaviside_comp import HeavisideComp
from fem2d.openmdao.penalization_comp import PenalizationComp
from fem2d.openmdao.states_comp import StatesComp
from fem2d.openmdao.disp_comp import DispComp
from fem2d.openmdao.compliance_comp import ComplianceComp
from fem2d.openmdao.weight_comp import WeightComp
from fem2d.openmdao.objective_comp import ObjectiveComp
from fem2d.fem2d import PyFEMSolver
from fem2d.utils.rbf import get_rbf_mtx
from fem2d.utils.bspline import get_bspline_mtx
from fem2d.utils.coords import get_coord_eval, get_coord_tmp


class FEM2DParamGroup(Group):

    def initialize(self):
        self.options.declare('fem_solver', types=PyFEMSolver, )#required=True)
        self.options.declare('length_x', types=(int, float), )#required=True)
        self.options.declare('length_y', types=(int, float), )#required=True)
        self.options.declare('num_nodes_x', types=int, )#required=True)
        self.options.declare('num_nodes_y', types=int, )#required=True)
        self.options.declare('num_param_x', types=int, )#required=True)
        self.options.declare('num_param_y', types=int, )#required=True)
        self.options.declare('forces', types=np.ndarray, )#required=True)
        self.options.declare('p', types=(int, float), )#required=True)
        self.options.declare('w', types=(int, float), )#required=True)
        self.options.declare('nodes', types=np.ndarray, )#required=True)
        self.options.declare('quad_order', default=4, types=int)
        self.options.declare('volume_fraction', types=(int, float), )#required=True)

    def setup(self):
        fem_solver = self.options['fem_solver']
        length_x = self.options['length_x']
        length_y = self.options['length_y']
        num_nodes_x = self.options['num_nodes_x']
        num_nodes_y = self.options['num_nodes_y']
        num_param_x = self.options['num_param_x']
        num_param_y = self.options['num_param_y']
        forces = self.options['forces']
        p = self.options['p']
        w = self.options['w']
        nodes = self.options['nodes']
        quad_order = self.options['quad_order']
        volume_fraction = self.options['volume_fraction']

        num = num_nodes_x * num_nodes_y

        coord_eval_x, coord_eval_y = get_coord_eval(num_nodes_x, num_nodes_y, quad_order)
        coord_eval_x *= length_x
        coord_eval_y *= length_y
        gpt_mesh = np.zeros((coord_eval_x.shape[0], coord_eval_y.shape[0], 2))
        gpt_mesh[:, :, 0] = np.outer(coord_eval_x, np.ones(coord_eval_y.shape[0]))
        gpt_mesh[:, :, 1] = np.outer(np.ones(coord_eval_x.shape[0]), coord_eval_y)

        state_size = 2 * num_nodes_x * num_nodes_y + 2 * num_nodes_y
        disp_size = 2 * num_nodes_x * num_nodes_y

        coord_eval_x, coord_eval_y = get_coord_eval(num_nodes_x, num_nodes_y, quad_order)

        if 1:
            param_mtx = get_bspline_mtx(
                coord_eval_x, coord_eval_y,
                num_param_x, num_param_y, kx=4, ky=4)
        else:
            param_mtx = get_rbf_mtx(
                coord_eval_x, coord_eval_y,
                num_param_x, num_param_y, kx=4, ky=4)

        rhs = np.zeros(state_size)
        rhs[:disp_size] = forces

        # inputs
        comp = IndepVarComp()
        comp.add_output('rhs', val=rhs)
        comp.add_output('forces', val=forces)

        # x = np.linalg.solve(
        #     param_mtx.T.dot(param_mtx),
        #     param_mtx.T.dot(0.5 * np.ones((num_nodes_x - 1) * (num_nodes_y - 1))))

        comp.add_output('dvs', val=0., shape=num_param_x * num_param_y)
        comp.add_design_var('dvs')
        self.add_subsystem('inputs_comp', comp)
        self.connect('inputs_comp.dvs', 'parametrization_comp.x')

        comp = ParametrizationComp(mtx=param_mtx,
            num_rows=(num_nodes_x - 1) * (num_nodes_y - 1) * quad_order ** 2,
            num_cols=num_param_x * num_param_y,
        )
        self.add_subsystem('parametrization_comp', comp)
        self.connect('parametrization_comp.y', 'states_comp.plot_var')

        # if 0:
        #     self.connect('parametrization_comp.y', 'averaging_comp.x')
        #     comp = AveragingComp(
        #         num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y, quad_order=quad_order)
        #     self.add_subsystem('averaging_comp', comp)
        #     self.connect('averaging_comp.y', 'heaviside_comp.x')
        #
        #     comp = HeavisideComp(num=num)
        #     self.add_subsystem('heaviside_comp', comp)
        #     self.connect('heaviside_comp.y', 'penalization_comp.x')
        #     self.connect('heaviside_comp.y', 'weight_comp.x')
        #
        #     comp = PenalizationComp(num=num, p=p)
        #     self.add_subsystem('penalization_comp', comp)
        #     self.connect('penalization_comp.y', 'states_comp.multipliers')

        if 1:
            self.connect('parametrization_comp.y', 'heaviside_comp.x')
            comp = HeavisideComp(num=(num_nodes_x - 1) * (num_nodes_y - 1) * quad_order ** 2)
            self.add_subsystem('heaviside_comp', comp)
            self.connect('heaviside_comp.y', 'averaging_comp.x')
            self.connect('heaviside_comp.y', 'states_comp.plot_var2')

            comp = AveragingComp(
                num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y, quad_order=quad_order)
            self.add_subsystem('averaging_comp', comp)
            self.connect('averaging_comp.y', 'penalization_comp.x')
            self.connect('averaging_comp.y', 'weight_comp.x')

            comp = PenalizationComp(num=num, p=p)
            self.add_subsystem('penalization_comp', comp)
            self.connect('penalization_comp.y', 'states_comp.multipliers')
        else:
            self.connect('parametrization_comp.y', 'heaviside_comp.x')
            comp = HeavisideComp(num=(num_nodes_x - 1) * (num_nodes_y - 1) * quad_order ** 2)
            self.add_subsystem('heaviside_comp', comp)
            self.connect('heaviside_comp.y', 'penalization_comp.x')
            self.connect('heaviside_comp.y', 'averaging_comp2.x')
            self.connect('heaviside_comp.y', 'states_comp.plot_var2')

            comp = AveragingComp(
                num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y, quad_order=quad_order)
            self.add_subsystem('averaging_comp2', comp)
            self.connect('averaging_comp2.y', 'weight_comp.x')

            comp = PenalizationComp(num=(num_nodes_x - 1) * (num_nodes_y - 1) * quad_order ** 2, p=p)
            self.add_subsystem('penalization_comp', comp)
            self.connect('penalization_comp.y', 'averaging_comp.x')

            comp = AveragingComp(
                num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y, quad_order=quad_order)
            self.add_subsystem('averaging_comp', comp)
            self.connect('averaging_comp.y', 'states_comp.multipliers')

        # states
        comp = StatesComp(
            fem_solver=fem_solver, num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y, nodes=nodes,
            gpt_mesh=gpt_mesh, quad_order=quad_order)
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
        comp = WeightComp(num=num)
        comp.add_constraint('weight', upper=volume_fraction)
        self.add_subsystem('weight_comp', comp)

        # objective
        comp = ObjectiveComp(w=w)
        comp.add_objective('objective')
        self.add_subsystem('objective_comp', comp)
        self.connect('compliance_comp.compliance', 'objective_comp.compliance')
        self.connect('weight_comp.weight', 'objective_comp.weight')
