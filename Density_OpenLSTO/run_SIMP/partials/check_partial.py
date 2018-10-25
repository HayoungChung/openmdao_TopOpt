import numpy as np

from openmdao.api import Group, ExplicitComponent, IndepVarComp, Problem

class MyComp(ExplicitComponent):
    def setup(self):
        self.add_input('x1', 3.0)
        self.add_input('x2', 5.0)

        self.add_output('y', 5.5)

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        
        outputs['y'] = 3.0*inputs['x1'] + 4.0*inputs['x2']

    def compute_partials(self, inputs, partials):
        
        J = partials
        J['y', 'x1'] = np.array([37.0])
        J['y', 'x2'] = np.array([40])

prob = Problem()
prob.model = Group()

prob.model.add_subsystem('p1', IndepVarComp('x1', 3.0))
prob.model.add_subsystem('p2', IndepVarComp('x2', 5.0))
prob.model.add_subsystem('comp', MyComp())

prob.model.connect('p1.x1', 'comp.x1')
prob.model.connect('p2.x2', 'comp.x2')

prob.set_solver_print(level=0)

prob.setup(check=False)
prob.run_model()
data = prob.check_partials()

x1_error = data['comp']['y', 'x1']['abs error']