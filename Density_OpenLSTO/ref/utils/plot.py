import numpy as np
import matplotlib.pylab as plt


def _plot(nodes, axes, color='k'):
    for ix in range(nodes.shape[0] - 1):
        for iy in range(nodes.shape[1] - 1):
            axes.plot(
                nodes[ix+0:ix+2, iy+0, 0],
                nodes[ix+0:ix+2, iy+0, 1],
                color)
            axes.plot(
                nodes[ix+0:ix+2, iy+1, 0],
                nodes[ix+0:ix+2, iy+1, 1],
                color)
            axes.plot(
                nodes[ix+0, iy+0:iy+2, 0],
                nodes[ix+0, iy+0:iy+2, 1],
                color)
            axes.plot(
                nodes[ix+1, iy+0:iy+2, 0],
                nodes[ix+1, iy+0:iy+2, 1],
                color)


def get_mesh(num_nodes_x, num_nodes_y, length_x, length_y):
    orig_nodes = np.zeros((num_nodes_x, num_nodes_y, 2))
    for ix in range(num_nodes_x):
        for iy in range(num_nodes_y):
            orig_nodes[ix, iy, 0] = ix / (num_nodes_x - 1) * length_x
            orig_nodes[ix, iy, 1] = iy / (num_nodes_y - 1) * length_y

    return orig_nodes


def get_gpt_mesh(num_nodes_x, num_nodes_y, length_x, length_y, quad_order):
    orig_nodes = np.zeros((num_nodes_x, num_nodes_y, 2))
    for ix in range(num_nodes_x - 1):
        for iy in range(num_nodes_y - 1):
            orig_nodes[ix, iy, 0] = ix / (num_nodes_x - 1) * length_x
            orig_nodes[ix, iy, 1] = iy / (num_nodes_y - 1) * length_y

    return orig_nodes


def plot_solution(orig_nodes, deflected_nodes=None):
    axes = plt.gca()
    _plot(orig_nodes, axes)
    if deflected_nodes is not None:
        _plot(deflected_nodes, axes, 'r')
    plt.show()


def plot_mesh(num_nodes_x, num_nodes_y, length_x, length_y):
    lins_x = np.linspace(0, length_x, num_nodes_x)
    lins_y = np.linspace(0, length_y, num_nodes_y)

    plt.subplot(2, 1, 1)

    for ix in range(num_nodes_x):
        plt.plot([lins_x[ix]] * 2, [0, length_y], 'grey', linewidth=0.4) #, zorder=1)

    for iy in range(num_nodes_y):
        plt.plot([0, length_x], [lins_y[iy]] * 2, 'grey', linewidth=0.4) #, zorder=1)

    plt.axis('equal')


def plot_contour(mesh, field, plot_boundary=False, plot_fill=False, plot_patch=False):
    if plot_fill:
        plt.subplot(2, 1, 1)
        x1 = np.min(mesh[:, :, 0])
        x2 = np.max(mesh[:, :, 0])
        y1 = np.min(mesh[:, :, 1])
        y2 = np.max(mesh[:, :, 1])
        plt.imshow(field.T, cmap='Greys', extent=[x1, x2, y1, y2]) #, zorder=2)

    if plot_boundary:
        length_x = mesh[-1, 0, 0]
        length_y = mesh[0, -1, 1]
        plt.subplot(2, 1, 2)
        plt.contour(mesh[:, ::-1, 0], mesh[:, ::-1, 1], field, levels=[0]) #, zorder=3)
        plt.plot([0] * 2, [0, length_y], 'k', linewidth=0.4) #, zorder=1)
        plt.plot([length_x] * 2, [0, length_y], 'k', linewidth=0.4) #, zorder=1)
        plt.plot([0, length_x], [0]*2, 'k', linewidth=0.4) #, zorder=1)
        plt.plot([0, length_x], [length_y]*2, 'k', linewidth=0.4) #, zorder=1)
        plt.axis('equal')
    
    if plot_patch:
        pass


def plot_save(save=None, show=False):
    ax = plt.subplot(2, 1, 1)
    plt.axis('equal')
    plt.axis('off')
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    ax = plt.subplot(2, 1, 2)
    plt.axis('off')
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])

    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()
    plt.clf()
