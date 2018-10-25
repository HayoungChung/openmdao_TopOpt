import os.path
try:
    import cPickle as pickle
except:
    import pickle

from plot import plot_contour, plot_save, plot_mesh


filename = 'const.pkl'
with open(filename, 'rb') as f:
    raw = pickle.load(f)

mesh = raw['mesh']

quad_order = raw['quad_order']
if quad_order is not None:
    gpt_mesh = raw['gpt_mesh']

length_x = 40#mesh[-1, 0, 0]
length_y = 20#mesh[0, -1, 1]
num_nodes_x = 41 #mesh.shape[0]
num_nodes_y = 21 #mesh.shape[1]

# raw['quad_order'] = quad_order
# raw['mesh'] = self.mesh
# raw['gpt_mesh'] = self.options['gpt_mesh']

counter = 0
filename = 'save/data%03i.pkl' % counter

while os.path.isfile(filename):
    print(counter)

    with open(filename, 'rb') as f:
        raw = pickle.load(f)

    # plot_mesh(num_nodes_x, num_nodes_y, length_x, length_y)

    if quad_order is not None:
        fill = raw['fill'].reshape((
            (num_nodes_x - 1) * quad_order,
            (num_nodes_y - 1) * quad_order,
        ),order = 'F',)
        boundary = raw['boundary'].reshape((
            (num_nodes_x - 1) * quad_order,
            (num_nodes_y - 1) * quad_order,
        ),order = 'F',)
        plot_contour(gpt_mesh, fill, plot_fill=True)
        plot_contour(gpt_mesh, boundary, plot_boundary=True)
    elif 0:
        multipliers = raw['multipliers'].reshape((num_nodes_x, num_nodes_y))
        plot_contour(mesh, multipliers, plot_fill=True)
    else:
        multipliers = raw['multipliers'].reshape((num_nodes_x-1, num_nodes_y-1),order='F').transpose()
        plot_contour(mesh, multipliers, plot_fill=True)

    plot_save(save='save/save%03i.png'%counter)

    counter += 1
    filename = 'save/data%03i.pkl' % counter

import movie
