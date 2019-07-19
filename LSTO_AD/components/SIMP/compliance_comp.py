import numpy as np

from openmdao.api import ExplicitComponent


class ComplianceComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes_x', types=int, )
        self.options.declare('num_nodes_y', types=int, )

    def setup(self):
        num_nodes_x = self.options['num_nodes_x']
        num_nodes_y = self.options['num_nodes_y']

        disp_size = 2 * num_nodes_x * num_nodes_y

        self.add_input('disp', shape=disp_size)
        self.add_input('forces', shape=disp_size)
        self.add_output('compliance')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['compliance'] = np.dot(inputs['disp'], inputs['forces'])

    def compute_partials(self, inputs, partials):
        partials['compliance', 'disp'][0, :] = inputs['forces']
        partials['compliance', 'forces'][0, :] = inputs['disp']

class HeatComplianceComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes_x', types=int, )
        self.options.declare('num_nodes_y', types=int, )

    def setup(self):
        num_nodes_x = self.options['num_nodes_x']
        num_nodes_y = self.options['num_nodes_y']

        disp_size = num_nodes_x * num_nodes_y

        self.add_input('disp', shape=disp_size)
        self.add_input('forces', shape=disp_size)
        self.add_output('compliance')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['compliance'] = np.dot(inputs['disp'], inputs['forces'])

    def compute_partials(self, inputs, partials):
        partials['compliance', 'disp'][0, :] = inputs['forces']
        partials['compliance', 'forces'][0, :] = inputs['disp']