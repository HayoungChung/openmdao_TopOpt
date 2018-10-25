import numpy as np

from fem2d.utils.gauss_quadrature import gauss_pts_dict


def get_coord_eval(num_nodes_x, num_nodes_y, quad_order, flatten=True):
    gauss_pts = gauss_pts_dict[quad_order]

    lins_x = np.linspace(0, 1, num_nodes_x)
    lins_y = np.linspace(0, 1, num_nodes_y)

    cx0 = (lins_x[1:] + lins_x[:-1]) / 2.
    cx1 = (lins_x[1:] - lins_x[:-1]) / 2.
    cy0 = (lins_y[1:] + lins_y[:-1]) / 2.
    cy1 = (lins_y[1:] - lins_y[:-1]) / 2.

    coord_eval_x = np.zeros((num_nodes_x - 1, quad_order))
    coord_eval_y = np.zeros((num_nodes_y - 1, quad_order))
    for ind in range(quad_order):
        g = gauss_pts[ind]
        coord_eval_x[:, ind] = cx0 + cx1 * g
        coord_eval_y[:, ind] = cy0 + cy1 * g

    if flatten:
        coord_eval_x = coord_eval_x.flatten()
        coord_eval_y = coord_eval_y.flatten()
    return coord_eval_x, coord_eval_y


def get_coord_tmp(num_cp_x, num_cp_y):
    num_tmp_x = 2 * num_cp_x
    num_tmp_y = 2 * num_cp_y

    coord_tmp = np.zeros((num_tmp_x, num_tmp_y, 2))

    lins_x = np.linspace(0, 1, num_tmp_x)
    lins_y = np.linspace(0, 1, num_tmp_y)
    ones_x = np.ones(num_tmp_x)
    ones_y = np.ones(num_tmp_y)

    coord_tmp[:, :, 0] = np.einsum('i,j->ij', lins_x, ones_y)
    coord_tmp[:, :, 1] = np.einsum('i,j->ij', ones_x, lins_y)

    return coord_tmp.reshape((num_tmp_x * num_tmp_y, 2))
