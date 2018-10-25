import numpy as np
from openmdao.api import ExplicitComponent
# from leastSquare_vec import leastSquare_vec # TODO

from lsm_classes import PyLSMSolver

# '''
# these are deprecated part (some preliminary works present, and results are meaningless
# '''
# class ComplSens_Comp(ExplicitComponent):
#     def initialize(self):
#         self.metadata.declare('lsm_solver', type_=PyLSMSolver, )#required=True)
#         self.metadata.declare('bpts_xy', type_=np.ndarray, )#required=True)
#         self.metadata.declare('fixedGpts_xy', type_=np.ndarray, )#required=True)
#         self.metadata.declare('radius', type_=(int, float), )#required=True)
#         self.metadata.declare('area_fraction', type_=np.ndarray, )#required=True)
#         self.metadata.declare('movelimit', type_=float, )#required=True)
#         # TOREMOVE
#         self.metadata.declare('tmpFix_bpts_sens', type_=np.ndarray, )#required=True)

#     def setup(self):
#         self.lsm_solver = self.metadata['lsm_solver']
#         self.bpts_xy = self.metadata['bpts_xy']
#         self.fixedGpts_xy = self.metadata['fixedGpts_xy']
#         self.radius = self.metadata['radius']
#         self.area_fraction = self.metadata['area_fraction']
#         self.movelimit = self.metadata['movelimit']
#         self.tmpFix_bpts_sens = self.metadata['tmpFix_bpts_sens']

#         num = fixedGpts_xy.shape[0]
#         self.add_input('fixedGpts_sens', shape=num, val=0.0)

#         self.add_output('bpts_sens', shape=num, val=0.0)
#         self.add_output('lowerbound', shape=np.ndarray, val=0.0)
#         self.add_output('upperbound', shape=np.ndarray, val=0.0)

#         self.declare_partials('bpts_sens','fixedGpts_sens',dependent=False)
#         self.declare_partials('lowerbound','fixedGpts_sens',dependent=False)
#         self.declare_partials('upperbound','fixedGpts_sens',dependent=False)

#     def compute(self, inputs, outputs):
#         # outputs['bpts_sens'] = leastSquare_vec(
#         #     lsm_inputs['fixedGpts_sens'], self.fixedGpts_sens,
#         #     self.radius, self.area_fraction)
#         outputs['bpts_sens'] = self.tmpFix_bpts_sens
#         lsm_solver.preprocess(self.movelimit, self.tmpFix_bpts_sens)
#         (outputs['upperbound'], outputs['lowerbound']) = lsm_solver.get_bounds()

# class AreaConstComp(ExplicitComponent):
#     def initialize(self):
#         self.metadata.declare('lsm_solver', type_=PyLSMSolver, )#required=True)
#         self.metadata.declare('num_bpts', type_=(int,float), )#required=True)
#         self.metadata.declare('num_dvs', type_=(int,float), )#required=True)

#     def setup(self):
#         self.lsm_solver = self.metadata['lsm_solver']
#         self.num_bpts = num_bpts = self.metadata['num_bpts']
#         num_dvs = self.metadata['num_dvs']

#         self.add_input('lambdas', shape = num_dvs, val = 0.)
#         self.add_output('bpts_sens', shape = (num_bpts,1))
#         self.add_output('G_cons', val = 0.)

#         # self.declare_partials('G_cons', 'lambas', dependent=True)
#         self.declare_partials('bpts_sens', 'lambdas', dependent=False)
#     def compute(self, inputs, outputs):
#         outputs['bpts_sens'] = -np.ones((self.num_bpts,1))
#         self.lsm_solver.computeDisplacements(inputs['lambdas'])
#         outputs['G_cons'] = self.lsm_solver.computeFunction(1)
#     def compute_partials(self, inputs, partials):
#         partials['G_cons','lambdas'] = self.lsm_solver.computeGradients(inputs['lambdas'], 1)

# class ScaleComp(ExplicitComponent):
#     def initialize(self):
#         self.metadata.declare('num_bpts', type_=int, )#required=True)
#         # self.metadata.declare('num_dvs', type_=int, )#required=True)

#     def setup(self):
#         self.num_bpts = num_bpts = self.metadata['num_bpts']
#         # num_dvs = self.metadata['num_dvs']
#         self.add_input('x', shape = num_bpts)

#         self.add_output('y', shape = (num_bpts,)) # num_dvs))
#         self.add_output('scale', shape = 1,)

#     def compute(self, inputs, outputs):
#         max_x = np.amax(np.abs((inputs['x'])))
#         mtx = np.identity(self.num_bpts) / max_x

#         outputs['y'] = mtx.dot(inputs['x'])
#         outputs['scale'] = 1.0/max_x


# class DispComp(ExplicitComponent):
#     def initialize(self):
#         self.metadata.declare('lsm_solver', type_=PyLSMSolver, )#required=True)
#         self.metadata.declare('num_bpts', type_=int, )#required=True)
#         self.metadata.declare('num_dvs', type_=int, )#required=True)
#         self.metadata.declare('segmentLength', type_=np.ndarray, )#required=True)

#     def setup(self):
#         self.lsm_solver = self.metadata['lsm_solver']
#         self.num_bpts = self.metadata['num_bpts']
#         self.num_dvs = self.metadata['num_dvs']
#         self.segmentLength = self.metadata['segmentLength']

#         self.add_input('sens_c', shape = self.num_bpts) #(self.num_bpts, self.num_dvs))
#         self.add_input('sens_a', shape = self.num_bpts)
#         self.add_input('lambdas', shape = self.num_dvs)
#         self.add_output('F_obj', val = 0.)

#         self.declare_partials('F_obj','sens_c', dependent=False)
#         self.declare_partials('F_obj','sens_a', dependent=False)

#     def compute(self, inputs, outputs):
#         self.sens = np.zeros((self.num_bpts, self.num_dvs))
#         self.sens[:,0] = inputs['sens_c']
#         self.sens[:,1] = inputs['sens_a']
#         for nn in range(0,self.num_dvs):
#             # outputs['F_obj'] += np.multiply(inputs['sens'][:,nn],self.segmentLength)*inputs['lambdas'][nn]
#             outputs['F_obj'] += sum(np.multiply(self.sens[:,nn],self.segmentLength)*inputs['lambdas'][nn])

#     def compute_partials(self,inputs, partials):
#         mtx = np.ndarray((1, self.num_dvs))
#         mtx[:,0] = sum(np.multiply(self.sens[:,0],self.segmentLength))
#         mtx[:,1] = sum(np.multiply(self.sens[:,1],self.segmentLength))
#         partials['F_obj', 'lambdas'][0] = mtx

# class UnscaleComp(ExplicitComponent):
#     def initialize(self):
#         pass
#     def setup(self):
#         self.add_input('')

# '''
# these are used for run_LSTO.py
# '''

class DisplacementComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('lsm_solver', type_=PyLSMSolver, )#required=True)
        self.metadata.declare('num_bpts', type_=int, )#required=True)
        self.metadata.declare('num_dvs', type_=int, )#required=True)

    def setup(self):
        self.lsm_solver = self.metadata['lsm_solver']
        num_bpts = self.metadata['num_bpts']
        num_dvs = self.metadata['num_dvs']
        self.add_input('lambdas', shape=2)
        self.add_output('displacements', shape=num_bpts)
        #self.approx_partials('displacements','lambdas',method='fd',form='central',step=1e-4)

        # outputs (temporary)
        # self.cnt = 0
        # (rows, cols, data) = self.lsm_solver.computePartialDisplacement()
        # self.declare_partials(of='displacements', wrt='lambdas', rows=rows, cols=cols)
        # self.declare_partials(of='displacements', wrt='lambdas', rows=rows, cols=cols, val=data)
    def compute(self, inputs, outputs):
        displacement = self.lsm_solver.computeDisplacements(inputs['lambdas'])
        outputs['displacements'] = np.asarray(displacement)


        # np.savetxt('txt_mdao/lambdas%d.txt'%self.cnt, inputs['lambdas'])
        # np.savetxt('txt_mdao/displacement_%d.txt'%self.cnt, np.asarray(displacement))
        # self.cnt += 1
    # def compute_partials(self, inputs, partials):
    #     self.lsm_solver.computeDisplacements(inputs['lambdas']) # displacement must be calculated first
    #     (rows, cols, data) = self.lsm_solver.computePartialDisplacement()
    #     partials['displacements', 'lambdas'] = data
    #     (rows, cols, data) = self.lsm_solver.computePartialDisplacement()
        
        # mat_pD = self.lsm_solver.computePartialDisplacement()
        # mat_pD_np = np.asarray(mat_pD)
        # print (pD.shape )
        # print (mat_pD_np.shape )
        # pD[:, 0] = mat_pD_np[:, 0]
        # pD[:, 1] = mat_pD_np[:, 1]


class ConstraintComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('lsm_solver', type_=PyLSMSolver, )#required=True)
        self.metadata.declare('num_bpts', type_=int, )#required=True)

    def setup(self):
        self.lsm_solver = self.metadata['lsm_solver']
        num_bpts = self.metadata['num_bpts']

        self.add_input('displacements', shape=num_bpts)
        self.add_output('constraint')
        # self.approx_partials('constraint', 'displacements')

        # outputs (temporary)
        # self.cnt = 0

    def compute(self, inputs, outputs):
        #outputs['constraint'] = self.lsm_solver.computeFunction(inputs['displacements'],1)[1]
        outputs['constraint'] = self.lsm_solver.computeFunction(
            inputs['displacements'], 1)

        # np.savetxt('txt_mdao/disp_cons%d.txt'%self.cnt, inputs['displacements'])
        # self.cnt += 1
    def compute_partials(self, inputs, partials):
        vec_ = self.lsm_solver.computePartialFunctions(1)
        partials['constraint','displacements'] = vec_

class ObjectiveComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('lsm_solver', type_=PyLSMSolver, )#required=True)
        self.metadata.declare('num_bpts', type_=int, )#required=True)

    def setup(self):
        self.lsm_solver = self.metadata['lsm_solver']
        num_bpts = self.metadata['num_bpts']

        self.add_input('displacements', shape=num_bpts)
        self.add_output('objective')
        # self.approx_partials('objective', 'displacements')

        # outputs (temporary)
        # self.cnt = 0

    def compute(self, inputs, outputs):
        outputs['objective'] = self.lsm_solver.computeFunction(
            inputs['displacements'], 0)

        # np.savetxt('txt_mdao/obj_cons%d.txt'%self.cnt, inputs['displacements'])
        # self.cnt += 1
    def compute_partials(self, inputs, partials):
        vec_ = self.lsm_solver.computePartialFunctions(0)
        partials['objective','displacements'] = vec_


class Callback_objF(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('lsm_solver', type_=PyLSMSolver, )#required=True)

    def setup(self):
        self.lsm_solver = self.metadata['lsm_solver']
        self.add_input('lambdas', shape=2)
        self.add_output('objective')

    def compute(self, inputs, outputs):
        #outputs['objective'] = self.lsm_solver.callback(inputs['lambdas'],0)
        displacement = self.lsm_solver.computeDisplacements(inputs['lambdas'])
        displacement_np = np.asarray(displacement)
        outputs['objective'] = self.lsm_solver.computeFunction(
            displacement_np, 0)


class Callback_conF(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('lsm_solver', type_=PyLSMSolver, )#required=True)

    def setup(self):
        self.lsm_solver = self.metadata['lsm_solver']
        self.add_input('lambdas', shape=2)
        self.add_output('constraint')
        self.approx_partials('constraint', 'lambdas')

    def compute(self, inputs, outputs):
        #outputs['constraint'] = self.lsm_solver.callback(inputs['lambdas'],1)
        displacement = self.lsm_solver.computeDisplacements(inputs['lambdas'])
        displacement_np = np.asarray(displacement)
        outputs['constraint'] = self.lsm_solver.computeFunction(displacement_np, 1)
