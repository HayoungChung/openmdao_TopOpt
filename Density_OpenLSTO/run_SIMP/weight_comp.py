import numpy as np

from openmdao.api import ExplicitComponent


class WeightComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num', types=int, )#required=True)
        self.options.declare('num_nodes_y', types=int, )#required=True)
        self.cnt = 0
    def setup(self):
        num = self.options['num']

        self.add_input('x', shape=num)
        self.add_output('weight')

        derivs = np.ones((1, num)) / num
        self.declare_partials('weight', 'x', val=derivs)

    def compute(self, inputs, outputs):

        outputs['weight'] = sum(inputs['x']) / self.options['num']
        print("[" + str(self.cnt) + "] weight: " + str(outputs['weight']))
        self.cnt += 1
