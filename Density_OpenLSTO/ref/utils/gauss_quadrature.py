from __future__ import division
import numpy as np
import numpy.polynomial.legendre


gauss_pts_dict = {
    1: np.array([
        0. ,
    ]),
    2: np.array([
        -np.sqrt(1/3) ,
        np.sqrt(1/3) ,
    ]),
    3: np.array([
        -np.sqrt(3/5) ,
        0. ,
        np.sqrt(3/5) ,
    ]),
    4: np.array([
        -np.sqrt(3/7 + 2/7 * np.sqrt(6/5)) ,
        -np.sqrt(3/7 - 2/7 * np.sqrt(6/5)) ,
        np.sqrt(3/7 - 2/7 * np.sqrt(6/5)) ,
        np.sqrt(3/7 + 2/7 * np.sqrt(6/5)) ,
    ]),
    5: np.array([
        -1/3 * np.sqrt(5 + 2 * np.sqrt(10/7)) ,
        -1/3 * np.sqrt(5 - 2 * np.sqrt(10/7)) ,
        0. ,
        1/3 * np.sqrt(5 - 2 * np.sqrt(10/7)) ,
        1/3 * np.sqrt(5 + 2 * np.sqrt(10/7)) ,
    ]),
}


gauss_wts_dict = {
    1: np.array([
        2. ,
    ]),
    2: np.array([
        1. ,
        1. ,
    ]),
    3: np.array([
        5/9 ,
        8/9 ,
        5/9 ,
    ]),
    4: np.array([
        (18 - np.sqrt(30)) / 36 ,
        (18 + np.sqrt(30)) / 36 ,
        (18 + np.sqrt(30)) / 36 ,
        (18 - np.sqrt(30)) / 36 ,
    ]),
    5: np.array([
        (322 - 13 * np.sqrt(70)) / 900 ,
        (322 + 13 * np.sqrt(70)) / 900 ,
        128/225 ,
        (322 + 13 * np.sqrt(70)) / 900 ,
        (322 - 13 * np.sqrt(70)) / 900 ,
    ]),
}


gauss_pts_dict = {}
gauss_wts_dict = {}
for num in range(1, 10):
    pts, wts = np.polynomial.legendre.leggauss(num)
    gauss_pts_dict[num] = pts
    gauss_wts_dict[num] = wts

if __name__ == '__main__':
    for num in range(1, 6):
        print(num)
        print('pts:', gauss_pts_dict[num])
        print('wts:', gauss_wts_dict[num])
        print(np.sum(gauss_wts_dict[num]))
        print(np.polynomial.legendre.leggauss(num))
        print()
