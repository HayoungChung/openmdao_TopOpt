import numpy as np
from openmdao.api import ExplicitComponent
from py_lsmBind import py_LSM
        
class DisplacementComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('nBpts', types=(int, float), )#required=True)
        self.options.declare('ndvs', types=(int, float), )#required=True)
        self.options.declare('lsm_solver', types=py_LSM, )#required=True) # considers geometry
        # self.options.declare('FreeNodes', types=np.ndarray, )#required=False)

    def setup(self):
        self.nBpts = self.options['nBpts'] 
        self.ndvs = self.options['ndvs'] 
        self.lsm_solver = self.options['lsm_solver']

        self.isActive = self.lsm_solver.get_isActive() # FreeNodes
        self.isBound = self.lsm_solver.get_isBound()  # DomainBoundaries

        self.add_input('lambdas', shape=self.ndvs)
        self.add_input('Scale_f')
        self.add_input('Scale_g',shape=self.ndvs-1)
        self.add_input('Sf', shape=self.nBpts)
        self.add_input('Sg', shape=(self.ndvs-1,self.nBpts))

        self.add_output('displacements', shape=self.nBpts)

        self.declare_partials('*','S*',dependent=False)
        self.declare_partials('*','lambdas', dependent=True, method ='fd')
        # self.approx_partials('displacements', 'lambdas', method='fd')

    def compute(self, inputs, outputs):
        lambdas = inputs['lambdas']
        Sf = inputs['Sf']
        Sg = inputs['Sg']

        Scale_f = inputs['Scale_f']
        Scale_g = inputs['Scale_g']

        # scaled displacements
        displacements = np.zeros(self.nBpts)

        (negLim, posLim) = self.lsm_solver.get_limits()

        for dd in range(self.nBpts):
            if (self.isActive[dd]):
                displacements[dd] += lambdas[0] * Sf[dd] * Scale_f
                for pp in range(self.ndvs-1):
                    displacements[dd] += lambdas[pp+1] * Sg[pp][dd] * Scale_g[pp]

                if self.isBound[dd]:
                    if displacements[dd] < negLim[dd]:
                        displacements[dd] = negLim[dd]

        outputs['displacements'] = displacements 

    # def compute_partials(self, in puts, partials):
    #     pass
        '''
        lambdas = inputs['lambdas']
        
        Sf = inputs['Sf']
        Sg = inputs['Sg']

        Scale_f = inputs['Scale_f']
        Scale_g = inputs['Scale_g']

        (negLim, posLim) = self.lsm_solver.get_limits()


        displacements = np.zeros(self.nBpts)
        disp_lambdas = np.zeros((self.nBpts, self.ndvs))

        for dd in range(self.nBpts):
            if (self.isActive[dd]):
                displacements[dd] += lambdas[0] * Sf[dd] * Scale_f
                disp_lambdas[dd,0] = Sf[dd] * Scale_f
                for pp in range(self.ndvs-1):
                    displacements[dd] += lambdas[pp+1] * Sg[pp][dd] * Scale_g[pp]
                    disp_lambdas[dd,pp] = Sg[pp][dd] * Scale_g[pp]

                if self.isBound[dd]:
                    if displacements[dd] < negLim[dd]:
                        disp_lambdas[dd,pp] = 0.0

        partials['displacements','lambdas'] = disp_lambdas
        '''
