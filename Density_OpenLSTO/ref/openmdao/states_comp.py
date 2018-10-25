import numpy as np
import scipy.sparse
import scipy.sparse.linalg

try:
    import cPickle as pickle
except:
    import pickle

from openmdao.api import ImplicitComponent

# from fem2d.fem2d import PyFEMSolver
from pyBind import py_FEA
from fem2d.utils.plot import plot_contour, plot_save, plot_mesh


class StatesComp(ImplicitComponent):

    def initialize(self):
        # self.options.declare('fem_solver', types=PyFEMSolver, )#required=True)
        self.options.declare('fem_solver', types=py_FEA, )#required=True)
        self.options.declare('num_nodes_x', types=int, )#required=True)
        self.options.declare('num_nodes_y', types=int, )#required=True)
        self.options.declare('nodes', types=np.ndarray, )#required=True)
        self.options.declare('gpt_mesh', types=np.ndarray, )#required=True)
        self.options.declare('quad_order', default=None, types=(int, type(None)))
        self.options.declare('isNodal', default=True, types=bool)

    def setup(self):
        fem_solver = self.options['fem_solver']
        num_nodes_x = self.options['num_nodes_x']
        num_nodes_y = self.options['num_nodes_y']
        quad_order = self.options['quad_order']
        isNodal = self.isNodal = self.options['isNodal']

        self.mesh = self.options['nodes']
        self.counter = 0

        state_size = 2 * num_nodes_x * num_nodes_y + 2 * num_nodes_y
        num_nodes = num_nodes_x * num_nodes_y
        num_elems = (num_nodes_x - 1) * (num_nodes_y - 1)
        
        if isNodal is True:
            self.add_input('multipliers', shape=num_nodes)
        else:
            self.add_input('multipliers', shape=num_elems)
        
        if quad_order is not None:
            self.add_input('plot_var', shape=(num_nodes_x - 1) * (num_nodes_y - 1) * quad_order ** 2)
            self.add_input('plot_var2', shape=(num_nodes_x - 1) * (num_nodes_y - 1) * quad_order ** 2)
        self.add_input('rhs', shape=state_size)
        self.add_output('states', shape=state_size)

        size = (num_nodes_x - 1) * (num_nodes_y - 1) * 64 * 4 + 2 * 2 * num_nodes_y
        self.data = data = np.zeros(size)
        self.rows = rows = np.zeros(size, np.int32)
        self.cols = cols = np.zeros(size, np.int32)

        fem_solver.get_stiffness_matrix(np.ones(num_nodes), data, rows, cols, self.isNodal)
        self.declare_partials('states', 'states', rows=rows, cols=cols) 

        size = (num_nodes_x - 1) * (num_nodes_y - 1) * 64 * 4
        self.data_d = data_d = np.zeros(size)
        self.rows_d = rows_d = np.zeros(size, np.int32)
        self.cols_d = cols_d = np.zeros(size, np.int32)

        fem_solver.get_stiffness_matrix_derivs(np.ones(state_size), data_d, rows_d, cols_d, self.isNodal)
        self.declare_partials('states', 'multipliers', rows=rows_d, cols=cols_d)

        data = -np.ones(state_size)
        arange = np.arange(state_size)
        self.declare_partials('states', 'rhs', val=data, rows=arange, cols=arange)

        if quad_order is not None:
            self.declare_partials('states', 'plot_var', dependent=False)
            self.declare_partials('states', 'plot_var2', dependent=False)

    def _get_mtx(self, inputs):
        fem_solver = self.options['fem_solver']
        num_nodes_x = self.options['num_nodes_x']
        num_nodes_y = self.options['num_nodes_y']

        state_size = 2 * num_nodes_x * num_nodes_y + 2 * num_nodes_y

        data, rows, cols = self.data, self.rows, self.cols

        fem_solver.get_stiffness_matrix(inputs['multipliers'], data, rows, cols, self.isNodal)

        mtx = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(state_size, state_size))

        return mtx

    def _compute_mtx_derivs(self, outputs):
        fem_solver = self.options['fem_solver']

        data_d, rows_d, cols_d = self.data_d, self.rows_d, self.cols_d

        fem_solver.get_stiffness_matrix_derivs(outputs['states'], data_d, rows_d, cols_d, self.isNodal)

    def _solve(self, sol, rhs, mode):
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
        
        # sol[:] = scipy.sparse.linalg.spsolve(self.mtx, rhs)
        # sol[:] = self.ilu.solve(rhs, arg)

    def apply_nonlinear(self, inputs, outputs, residuals):
        self.mtx = self._get_mtx(inputs)

        residuals['states'] = self.mtx.dot(outputs['states']) - inputs['rhs']

    def solve_nonlinear(self, inputs, outputs):
        self.mtx = self._get_mtx(inputs)
        self.ilu = scipy.sparse.linalg.spilu(self.mtx, drop_tol=1e-14)

        self._solve(outputs['states'], inputs['rhs'], 'fwd')
        self._compute_mtx_derivs(outputs)

    def linearize(self, inputs, outputs, partials):
        num_nodes_x = self.options['num_nodes_x']
        num_nodes_y = self.options['num_nodes_y']
        quad_order = self.options['quad_order']

        self.mtx = self._get_mtx(inputs)
        self._compute_mtx_derivs(outputs)

        partials['states', 'states'] = self.data
        partials['states', 'multipliers'] = self.data_d

        self.ilu = scipy.sparse.linalg.spilu(self.mtx, drop_tol=1e-14)

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
        if mode == 'fwd':
            self._solve(d_outputs['states'], d_residuals['states'], 'fwd')
        elif mode == 'rev':
            self._solve(d_residuals['states'], d_outputs['states'], 'rev')
