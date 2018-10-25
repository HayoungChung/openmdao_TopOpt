import numpy as np

from openmdao.api import ExplicitComponent


class HeavisideComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num', types=int, )#required=True)

    def setup(self):
        num = self.options['num']

        self.add_input('x', shape=num)
        self.add_output('y', shape=num)

        arange = np.arange(num)
        self.declare_partials('y', 'x', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs['y'] = 0.5 + 0.5 * np.tanh(inputs['x']) + 1e-5

    def compute_partials(self, inputs, partials):
        partials['y', 'x'] = 0.5 / np.cosh(inputs['x']) ** 2
