`import numpy as np
import scipy.sparse

from openmdao.api import ExplicitComponent

from fem2d.utils.gauss_quadrature import gauss_wts_dict, gauss_pts_dict


class AveragingComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes_x', types=int, )#required=True)
        self.options.declare('num_nodes_y', types=int, )#required=True)
        self.options.declare('quad_order', types=int, )#required=True)

    def setup(self):
        nx = self.options['num_nodes_x']
        ny = self.options['num_nodes_y']
        quad_order = self.options['quad_order']

        gauss_wts = gauss_wts_dict[quad_order] / 2.
        gauss_pts = gauss_pts_dict[quad_order] / 2.

        self.add_input('x', shape=(nx - 1) * quad_order * (ny - 1) * quad_order)
        self.add_output('y', shape=nx * ny)

        x_arange = np.arange((nx - 1) * quad_order * (ny - 1) * quad_order).reshape(
            (nx - 1, quad_order, ny - 1, quad_order))
        y_arange = np.arange(nx * ny).reshape((nx, ny))

        data = np.zeros((nx - 1, quad_order, ny - 1, quad_order, 4))
        rows = np.zeros((nx - 1, quad_order, ny - 1, quad_order, 4), int)
        cols = np.zeros((nx - 1, quad_order, ny - 1, quad_order, 4), int)

        data[:, :, :, :, 0] = np.einsum('ik,j,l->ijkl', np.ones((nx - 1, ny - 1)),
            gauss_wts * (0.5 - gauss_pts), gauss_wts * (0.5 - gauss_pts))
        data[:, :, :, :, 1] = np.einsum('ik,j,l->ijkl', np.ones((nx - 1, ny - 1)),
            gauss_wts * (0.5 + gauss_pts), gauss_wts * (0.5 - gauss_pts))
        data[:, :, :, :, 2] = np.einsum('ik,j,l->ijkl', np.ones((nx - 1, ny - 1)),
            gauss_wts * (0.5 + gauss_pts), gauss_wts * (0.5 + gauss_pts))
        data[:, :, :, :, 3] = np.einsum('ik,j,l->ijkl', np.ones((nx - 1, ny - 1)),
            gauss_wts * (0.5 - gauss_pts), gauss_wts * (0.5 + gauss_pts))

        rows[:, :, :, :, 0] = np.einsum('ik,jl->ijkl',
            y_arange[:-1, :-1], np.ones((quad_order, quad_order), int))
        rows[:, :, :, :, 1] = np.einsum('ik,jl->ijkl',
            y_arange[1:, :-1], np.ones((quad_order, quad_order), int))
        rows[:, :, :, :, 2] = np.einsum('ik,jl->ijkl',
            y_arange[1:, 1:], np.ones((quad_order, quad_order), int))
        rows[:, :, :, :, 3] = np.einsum('ik,jl->ijkl',
            y_arange[:-1, 1:], np.ones((quad_order, quad_order), int))

        cols[:, :, :, :, 0] = x_arange
        cols[:, :, :, :, 1] = x_arange
        cols[:, :, :, :, 2] = x_arange
        cols[:, :, :, :, 3] = x_arange

        data = data.flatten()
        rows = rows.flatten()
        cols = cols.flatten()

        self.declare_partials('y', 'x', val=data, rows=rows, cols=cols)

        self.mtx = scipy.sparse.csc_matrix((data, (rows, cols)),
            shape=(nx * ny, (nx - 1) * quad_order * (ny - 1) * quad_order))

    def compute(self, inputs, outputs):
        outputs['y'] = self.mtx.dot(inputs['x'])
