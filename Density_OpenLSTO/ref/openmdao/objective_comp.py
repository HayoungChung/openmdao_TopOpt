import numpy as np

from openmdao.api import ExplicitComponent


class ObjectiveComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('w', types=(int, float), )#required=True)

    def setup(self):
        w = self.options['w']

        self.add_input('weight')
        self.add_input('compliance')
        self.add_output('objective')
        self.declare_partials('objective', 'weight', val=w)
        self.declare_partials('objective', 'compliance', val=1-w)

    def compute(self, inputs, outputs):
        w = self.options['w']

        outputs['objective'] = w * inputs['weight'] + (1 - w) * inputs['compliance']
        
        # save
        f = open('./save/weight_compl', 'a')
        f.write('%5.8f %5.8f\n' % (inputs['weight'], inputs['compliance'])) 
        f.close