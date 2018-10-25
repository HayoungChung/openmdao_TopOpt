import numpy as np
from openmdao.api import ExplicitComponent
from py_lsmBind import py_LSM
        
class ScalingComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('nBpts', types=(int,float), )#required=True)
        self.options.declare('lsm_solver', types=py_LSM, )#required=True)

    def setup(self):
        self.nBpts = self.options['nBpts']
        lsm_solver = self.options['lsm_solver']
        self.isActive = lsm_solver.get_isActive()

        self.add_input('x', shape=self.nBpts)
        self.add_output('y', shape = 1)
        self.declare_partials('y','x',dependent=False)

    def compute(self, inputs, outputs):
        x = inputs['x']

        maxSens = abs(x)
        maxSens[self.isActive] = 0.0

        self.ind = np.argmax(maxSens)
        outputs['y'] = 1.0/max(maxSens)

                
    # def compute_partials(self, inputs, partials):
    #     pass
        # x = inputs['x']

        # y_x = np.zeros(self.nBpts)
        # y_x[self.ind] = np.sign(x[self.ind])
        
        # partials['y','x'] = y_x
