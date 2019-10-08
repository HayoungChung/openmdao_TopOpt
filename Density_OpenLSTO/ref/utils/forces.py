import numpy as np


def get_forces(num_nodes_x, num_nodes_y, f=-10):
    vecF = np.zeros(2*num_nodes_x*num_nodes_y)
    vecF[(num_nodes_y*(num_nodes_x-1)+int(num_nodes_y/2))*2+1] = f

    return vecF
