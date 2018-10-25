import numpy as np
from openmdao.api import ExplicitComponent
from py_lsmBind import py_LSM

class IntegralComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('lsm_solver', type_=py_LSM, )#required=True)
        self.metadata.declare('nBpts', type_=(int,float), )#required=True)

    def setup(self):
        self.nBpts = self.metadata['nBpts']
        self.lsm_solver = self.metadata['lsm_solver']

        self.segLength = self.lsm_solver.get_length()

        self.add_input('x', shape=self.nBpts) 
        self.add_output('y', shape=self.nBpts)

        ind = np.arange(self.nBpts)
        self.declare_partials('y','x',rows=ind, cols=ind, val = self.segLength) 

    def compute(self, inputs, outputs):
        intVal = np.zeros(self.nBpts)

        for ii in range(self.nBpts):
            intVal[ii] = inputs['x'][ii] * self.segLength[ii]

        outputs['y'] = intVal


    def compute_partials(self, inputs, partials):
        pass