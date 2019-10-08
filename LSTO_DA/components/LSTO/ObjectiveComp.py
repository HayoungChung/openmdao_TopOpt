import numpy as np
from openmdao.api import ExplicitComponent
from py_lsmBind import py_LSM

class ObjectiveComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('nBpts', types=(int, float), )#required=True)
        self.options.declare('lsm_solver', types=py_LSM, )#required=True)

    def setup(self):
        self.nBpts = self.options['nBpts'] 
        self.lsm_solver = self.options['lsm_solver']
        self.isActive = self.lsm_solver.get_isActive()
        self.isBound = self.lsm_solver.get_isBound()
        
        self.add_input('scale_f')
        self.add_input('Cf', shape=self.nBpts)
        self.add_input('displacements', shape=self.nBpts)

        self.add_output('delF', shape=1)

        self.declare_partials('delF', 'scale_f', dependent=False)
        self.declare_partials('delF', 'Cf', dependent=False)
        self.declare_partials('delF', 'displacements', dependent=True)

    def compute(self, inputs, outputs):
        scale_f = inputs['scale_f']
        Cf = inputs['Cf']
        displacements = inputs['displacements']

        func = 0.
        for dd in range(self.nBpts):
            if self.isActive[dd]:
                func += scale_f * displacements[dd] * Cf[dd]
        outputs['delF'] =func 
        # print ('delF = ')
        # print outputs['delF']

    def compute_partials(self, inputs, partials):
        scale_f = inputs['scale_f']
        Cf = inputs['Cf']

        func = np.zeros(self.nBpts)

        for dd in range(self.nBpts):
            if self.isActive[dd]:
                func[dd] = scale_f * Cf[dd]

        partials['delF','displacements'] = func 
