import numpy as np

from openmdao.api import ExplicitComponent


class ParametrizationComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_rows', types=int, )#required=True)
        self.options.declare('num_cols', types=int, )#required=True)
        self.options.declare('mtx', )#required=True)

    def setup(self):
        num_rows = self.options['num_rows']
        num_cols = self.options['num_cols']
        mtx = self.options['mtx']

        self.add_input('x', shape=num_cols)
        self.add_output('y', shape=num_rows)
        self.declare_partials('y', 'x', val=mtx)

    def compute(self, inputs, outputs):
        outputs['y'] = self.options['mtx'].dot(inputs['x'])
