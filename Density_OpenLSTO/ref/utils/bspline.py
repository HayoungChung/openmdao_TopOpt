import numpy as np
from scipy.interpolate import bisplrep, bisplev, RectBivariateSpline

from fem2d.utils.coords import get_coord_tmp


def get_bspline_mtx(coord_eval_x, coord_eval_y, num_cp_x, num_cp_y, kx=4, ky=4):
    coord_tmp = get_coord_tmp(num_cp_x, num_cp_y)

    num_eval_x = coord_eval_x.shape[0]
    num_eval_y = coord_eval_y.shape[0]

    num_tmp = coord_tmp.shape[0]
    num_eval = num_eval_x * num_eval_y
    num_cp = num_cp_x * num_cp_y

    tx = np.linspace(0, 1, num_cp_x + kx + 1)
    ty = np.linspace(0, 1, num_cp_y + ky + 1)
    nxest = num_cp_x + kx + 1
    nyest = num_cp_y + ky + 1

    tmp = np.ones(num_tmp)
    tck = bisplrep(
        coord_tmp[:, 0], coord_tmp[:, 1], coord_tmp[:, 0],
        task=-1, kx=kx, ky=ky, tx=tx, ty=ty, nxest=nxest, nyest=nyest,
        xb=0., xe=1., yb=0., ye=1.)

    h = 1e-3
    mtx = np.zeros((num_eval, num_cp))
    out0 = bisplev(coord_eval_x, coord_eval_y, tck).flatten()
    for ind in range(num_cp):
        tck[2][ind] += h
        out = bisplev(coord_eval_x, coord_eval_y, tck).flatten()
        tck[2][ind] -= h
        mtx[:, ind] = (out - out0) / h

    return mtx
