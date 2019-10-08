import numpy as np
from openmdao.api import ExplicitComponent
from py_lsmBind import py_LSM

class ConstraintComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('nBpts', types=(int, float), )#required=True)
        self.options.declare('lsm_solver', types=py_LSM, )#required=True)

    def setup(self):
        self.nBpts = self.options['nBpts'] 
        self.lsm_solver = self.options['lsm_solver']
        self.isActive = self.lsm_solver.get_isActive()
        self.isBound = self.lsm_solver.get_isBound()
        
        self.add_input('scale_g')
        self.add_input('Cg', shape=self.nBpts)
        self.add_input('constraintDistance')
        self.add_input('displacements', shape=self.nBpts)

        self.add_output('delG')

        self.declare_partials('delG', 'scale_g', dependent=False)
        self.declare_partials('delG', 'Cg', dependent=False)
        self.declare_partials('delG', 'constraintDistance', dependent=False)
        self.declare_partials('delG', 'displacements', dependent=True)

    def compute(self, inputs, outputs):
        scale_g = inputs['scale_g']
        Cg = inputs['Cg']
        displacements = inputs['displacements']
        constraintDistance = inputs['constraintDistance']

        # scaled_constraintDistance = self.lsm_solver._compute_scaledConstraintDistance(constraintDistance, Cg)
        
        scaled_constraintDistance = self.lsm_solver.compute_scaledConstraintDistance(constraintDistance)
        # print (scaled_constraintDistance, scaled_constraintDistance2)

        func = 0.
        for dd in range(self.nBpts):
            if self.isActive[dd]:
                func += scale_g * displacements[dd] * Cg[dd]
                
        outputs['delG'] = func - scaled_constraintDistance * scale_g
        # print ('delG = ')
        # print outputs['delG']

    def compute_partials(self, inputs, partials):
        scale_g = inputs['scale_g']
        Cg = inputs['Cg']
        constraintDistance = inputs['constraintDistance']

        func = np.zeros(self.nBpts)
        for dd in range(self.nBpts):
            if self.isActive[dd]:
                func[dd] = scale_g * Cg[dd]
                
        partials['delG','displacements'] = func
