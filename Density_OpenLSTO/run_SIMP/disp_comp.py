import numpy as np

from openmdao.api import ExplicitComponent


class DispComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('nDOF', types=int, )#required=True)
        self.options.declare('nSTATE', types=int, )#required=True)

    def setup(self):
        nDOF = self.options['nDOF']
        nSTATE = self.options['nSTATE']

        self.add_input('states', shape=nSTATE)
        self.add_output('disp', shape=nDOF)

        self.declare_partials('disp', 'states',
            val=np.ones(nDOF), rows=np.arange(nDOF), cols=np.arange(nDOF))

    def compute(self, inputs, outputs):
        nDOF = self.options['nDOF']

        outputs['disp'] = inputs['states'][:nDOF]
