import numpy as np
import os.path
try:
    import cPickle as pickle
except:
    import pickle

# from plot import plot_contour, plot_save, plot_mesh

import matplotlib
import matplotlib.pyplot as plt


filename = 'const.pkl'
with open(filename, 'rb') as f:
    raw = pickle.load(f)

mesh = raw['mesh']

length_x = mesh[-1, 0, 0]
length_y = mesh[0, -1, 1]
num_nodes_x = mesh.shape[0]
num_nodes_y = mesh.shape[1]


counter = 0
filename = 'save/data%03i.pkl' % counter

with open(filename, 'rb') as f:
    raw = pickle.load(f)

    # plot_mesh(num_nodes_x, num_nodes_y, length_x, length_y)

    lins_x = np.linspace(0, length_x, num_nodes_x)
    lins_y = np.linspace(0, length_y, num_nodes_y)

    # plt.subplot(2, 1, 1)

    for ix in range(num_nodes_x):
        plt.plot([lins_x[ix]] * 2, [0, length_y], 'grey', linewidth=0.4) #, zorder=1)

    # for iy in range(num_nodes_y):
    #     plt.plot([0, length_x], [lins_y[iy]] * 2, 'grey', linewidth=0.4) #, zorder=1)

    # plt.axis('equal')

    # ax = plt.subplot(2, 1, 1)
    # plt.axis('equal')
    # plt.axis('off')
    # xmin,xmax = ax.get_xlim()
    # ymin,ymax = ax.get_ylim()
    # ax = plt.subplot(2, 1, 2)
    # plt.axis('off')
    # ax.set_xlim([xmin,xmax])
    # ax.set_ylim([ymin,ymax])

    # # if save is not None:
    # #     plt.savefig(save)
    # # if show:
    # plt.show()
    # plt.clf()

