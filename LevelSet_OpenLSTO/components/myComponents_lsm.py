import numpy as np
from openmdao.api import ExplicitComponent
sys.path.insert(0, r'/home/hac210/Dropbox/packages/02.M2DO_opensource_new/OpenLSTO/M2DO_LSM/Python/')
from py_lsmBind import py_LSM
        

class IntegralComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('lsm_solver', type_=py_LSM, )#required=True)

    def setup(self):
        self.add_input('Sf', shape=num, val = 0.0)
        self.add_input('Sg', shape=num, val = 0.0)
        self.add_input('segLength', shape=np.ndarray, val =0.0)
         
        self.add_output('Cf', shape=num, val = 0.0)
        self.add_output('Cg', shape=num, val = 0.0)

    def compute(self, inputs, outputs):
        outputs['Cf'] = 
    def compute_partials(self, inputs, partials):
        pass


