import numpy as np


def get_rbf_mtx(coord_eval_x, coord_eval_y, num_param_x, num_param_y, kx=4, ky=4):
    num_eval_x = coord_eval_x.shape[0]
    num_eval_y = coord_eval_y.shape[0]

    num_eval = num_eval_x * num_eval_y
    num_param = num_param_x * num_param_y

    # ----- ----- ----- ----- ----- ----- -----

    coord_eval = np.zeros((num_eval_x, num_eval_y, 2))
    coord_param = np.zeros((num_param_x, num_param_y, 2))

    coord_eval[:, :, 0] = np.einsum('i,j->ij',
        np.linspace(0, 1, num_eval_x), np.ones(num_eval_y))
    coord_eval[:, :, 1] = np.einsum('i,j->ij',
        np.ones(num_eval_x), np.linspace(0, 1, num_eval_y))

    coord_param[:, :, 0] = np.einsum('i,j->ij',
        np.linspace(0, 1, num_param_x), np.ones(num_param_y))
    coord_param[:, :, 1] = np.einsum('i,j->ij',
        np.ones(num_param_x), np.linspace(0, 1, num_param_y))

    coord_eval = coord_eval.reshape((num_eval_x * num_eval_y, 2))
    coord_param = coord_param.reshape((num_param_x * num_param_y, 2))

    # ----- ----- ----- ----- ----- ----- -----

    grid_eval = np.zeros((num_eval, num_param, 2))
    grid_param = np.zeros((num_eval, num_param, 2))

    for ind in range(2):
        grid_eval[:, :, ind] = np.einsum('i,j->ij', coord_eval[:, ind], np.ones(num_param))
        grid_param[:, :, ind] = np.einsum('i,j->ij', np.ones(num_eval), coord_param[:, ind])

    diff_sq = (grid_eval - grid_param) ** 2
    r_sq = diff_sq[:, :, 0] + diff_sq[:, :, 1]

    phi_mtx = np.exp(-r_sq)
    phi_mtx = np.sqrt(r_sq + 0.01 ** 2)

    return phi_mtx
