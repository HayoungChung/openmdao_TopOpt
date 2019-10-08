import numpy as np

from openmdao.api import ExplicitComponent


class SumComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('w', types=float,) # weight

    def setup(self):
        w = self.options['w']

        self.add_input('x1')#, shape=1)
        self.add_input('x2')#, shape=1)
        self.add_output('y')

        self.declare_partials('y', 'x1', val=w)
        self.declare_partials('y', 'x2', val=1.-w)

    def compute(self, inputs, outputs):
        w = self.options['w']
        outputs['y'] = w*inputs['x1'] + (1.-w)*inputs['x2']
