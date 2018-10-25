import numpy as np
import scipy.sparse
import scipy.sparse.linalg

try:
    import cPickle as pickle
except:
    import pickle

from openmdao.api import ImplicitComponent

from pyBind import py_FEA
# from fem2d.fem2d import PyFEMSolver
# from fem2d.utils.plot import plot_contour, plot_save, plot_mesh

import matplotlib.pyplot as plt

class StatesComp(ImplicitComponent):

    def initialize(self):
        self.options.declare('fem_solver', types=py_FEA, )#required=True)
        self.options.declare('num_nodes_x', types=int, )#required=True)
        self.options.declare('num_nodes_y', types=int, )#required=True)
        self.options.declare('BCid', types=np.ndarray, )#required=True)        
        self.options.declare('nodes', types=np.ndarray, )#required=True)
        self.options.declare('gpt_mesh', types=np.ndarray, )#required=True)
        self.options.declare('quad_order', default=None, types=(int, type(None)))
        self.options.declare('isNodal', default=True, types=bool)


    def setup(self):
        fem_solver = self.options['fem_solver']
        num_nodes_x = self.options['num_nodes_x']
        num_nodes_y = self.options['num_nodes_y']
        quad_order = self.options['quad_order']
        self.BCid = BCid = self.options['BCid']
        isNodal = self.isNodal = self.options['isNodal']

        self.mesh = self.options['nodes']
        self.counter = 0

        # state_size = 2 * num_nodes_x * num_nodes_y + 2 * num_nodes_y
        self.num_nodes = num_nodes = num_nodes_x * num_nodes_y
        self.num_elems = num_elems = (num_nodes_x - 1) * (num_nodes_y - 1)
        self.num_dofs = num_dofs = num_nodes * 2
        self.num_bc   = num_bc = BCid.size
        self.num_states = num_states = num_dofs + num_bc
        
        if isNodal is True:
            self.add_input('multipliers', shape=num_nodes) # param
        else:
            self.add_input('multipliers', shape=num_elems) # simp
        
        if quad_order is not None:
            ''' for param_LSTO: communicating heaviside parameter '''
            self.add_input('plot_var', shape=(num_nodes_x - 1) * (num_nodes_y - 1) * quad_order ** 2)
            self.add_input('plot_var2', shape=(num_nodes_x - 1) * (num_nodes_y - 1) * quad_order ** 2)
            
        self.add_input('rhs', shape=num_states)
        self.add_output('states', shape=num_states)

        (rows, cols, vals) = fem_solver.compute_K_SIMP(np.ones(num_elems))
        # npvals = np.array(vals)
        # nprows = np.array(rows, dtype=np.int32)
        # npcols = np.array(cols, dtype=np.int32)

        Ifix = self.num_dofs + np.arange(self.num_bc)
        nprows = np.append(rows, [Ifix, self.BCid])
        npcols = np.append(cols, [self.BCid, Ifix])
        # npvals = np.append(vals, [np.ones(self.num_bc*2)])

        self.declare_partials('states', 'states', rows=nprows, cols=npcols) 
            
        (rows_d, cols_d, vals_d) = fem_solver.compute_K_SIMP_derivs(np.ones(num_states)) 
        npvals_d = np.array(vals_d, dtype=float)
        nprows_d = np.array(rows_d, dtype=np.int32)
        npcols_d = np.array(cols_d, dtype=np.int32)
        self.declare_partials('states', 'multipliers', rows=nprows_d, cols=npcols_d)

        data = -np.ones(num_states)
        arange = np.arange(num_states)
        self.declare_partials('states', 'rhs', val=data, rows=arange, cols=arange)

        if quad_order is not None:
            self.declare_partials('states', 'plot_var', dependent=False)
            self.declare_partials('states', 'plot_var2', dependent=False)

    def _get_mtx(self, inputs):
        fem_solver = self.options['fem_solver']
        (rows, cols, vals) = fem_solver.compute_K_SIMP(inputs['multipliers'])

        # npvals = np.array(vals)
        # nprows = np.array(rows, dtype=np.int32)
        # npcols = np.array(cols, dtype=np.int32)

        Ifix = self.num_dofs + np.arange(self.num_bc)
        nprows = np.append(rows, [Ifix, self.BCid])
        npcols = np.append(cols, [self.BCid, Ifix])
        npvals = np.append(vals, [np.ones(self.num_bc*2)])

        mtx = scipy.sparse.csc_matrix((npvals, (nprows, npcols)), shape=(self.num_states, self.num_states))
        return (mtx, npvals)

    def _compute_mtx_derivs(self, outputs):
        fem_solver = self.options['fem_solver']
        (rows, cols, vals) = fem_solver.compute_K_SIMP_derivs(outputs['states'])

        return np.array(vals)

    def _solve(self, sol, rhs, mode='fwd'):
        size = sol.shape[0]
        # sol[:] = scipy.sparse.linalg.spsolve(self.mtx.T, rhs) 
        # fem_solver = self.options['fem_solver']
        # sol[:] = np.array(fem_solver.solve_FE())

        if mode == 'fwd':
            arg = 'N'
        elif mode == 'rev':
            arg = 'T'

        class Callback(object):
            def __init__(self, mtx):
                self.counter = 0
                self.mtx = mtx
            def __call__(self, xk):
                # print('%3i ' % self.counter, np.linalg.norm(self.mtx.dot(xk) - rhs))
                # print('%3i ' % self.counter, np.linalg.norm(xk))
                self.counter += 1

        class PC(object):
            def __init__(self, arg, ilu):
                self.arg = arg
                self.ilu = ilu
            def __call__(self, rhs):
                return self.ilu.solve(rhs, self.arg)

        size = sol.shape[0]
        pc_op = scipy.sparse.linalg.LinearOperator((size, size), matvec=PC(arg, self.ilu))
        sol[:] = scipy.sparse.linalg.gmres(
            self.mtx.T, rhs, x0=sol, M=pc_op,
            callback=Callback(self.mtx), tol=1e-10, restart=200,
        )[0]

    def apply_nonlinear(self, inputs, outputs, residuals):
        self.mtx = self._get_mtx(inputs)[0]
        # print("apply_nonlinear")
        residuals['states'] = self.mtx.dot(outputs['states']) - inputs['rhs']

    def solve_nonlinear(self, inputs, outputs):
        self.mtx = self._get_mtx(inputs)[0]
        self.ilu = scipy.sparse.linalg.spilu(self.mtx, drop_tol=1e-14)

        # outputs['states'] = scipy.sparse.linalg.spsolve(self.mtx, inputs['rhs']) # direct solver is used
        #print("solve_nonlinear")
        self._solve(outputs['states'], inputs['rhs'], 'fwd') 
        # self._compute_mtx_derivs(outputs) # ?

    def linearize(self, inputs, outputs, partials):
        #print("linearize")
        num_nodes_x = self.options['num_nodes_x']
        num_nodes_y = self.options['num_nodes_y']
        quad_order = self.options['quad_order']

        (self.mtx, Kij) = self._get_mtx(inputs)
        Kij_deriv = self._compute_mtx_derivs(outputs) 

        partials['states', 'states'] = Kij
        partials['states', 'multipliers'] = Kij_deriv

        self.ilu = scipy.sparse.linalg.spilu(self.mtx, drop_tol=1e-14) # ? 


        if self.counter == 0:
            raw = {}
            raw['quad_order'] = quad_order
            raw['mesh'] = self.mesh
            if quad_order is not None:
                raw['gpt_mesh'] = self.options['gpt_mesh']
            filename = 'const.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(raw, f)

        raw = {}
        if quad_order is not None:
            raw['fill'] = inputs['plot_var2']
            raw['boundary'] = inputs['plot_var']
        else:
            raw['multipliers'] = inputs['multipliers']
        filename = 'save/data%03i.pkl' % self.counter
        with open(filename, 'wb') as f:
            pickle.dump(raw, f)

        self.counter += 1

    def solve_linear(self, d_outputs, d_residuals, mode):
        #print("solve_nonlinear")
        if mode == 'fwd':
            self._solve(d_outputs['states'], d_residuals['states'], 'fwd')
        elif mode == 'rev':
            self._solve(d_residuals['states'], d_outputs['states'], 'rev')
