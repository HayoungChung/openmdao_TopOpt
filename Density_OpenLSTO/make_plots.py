import os.path
try:
    import cPickle as pickle
except:
    import pickle

import numpy as np 
from utils.plot import plot_contour, plot_save, plot_mesh


filename = 'const.pkl'
with open(filename, 'rb') as f:
    raw = pickle.load(f)

lxy  = raw['lxy']
exy  = raw['exy']

mesh = raw['node']
elem = raw['elem']
quad_order = raw['quad_order']

if quad_order is not None:
    gpt_mesh = raw['gpt_mesh']

length_x = lxy[0]
length_y = lxy[1]

num_nodes_x = exy[0] + 1
num_nodes_y = exy[1] + 1 
# raw['quad_order'] = quad_order
# raw['mesh'] = self.mesh
# raw['gpt_mesh'] = self.options['gpt_mesh']

counter = 0
filename = 'save/data%03i.pkl' % counter

while os.path.isfile(filename):
    print(counter)

    with open(filename, 'rb') as f:
        raw = pickle.load(f)

    plot_mesh(num_nodes_x, num_nodes_y, length_x, length_y)

    if quad_order is not None:
        fill = raw['fill'].reshape((
            (num_nodes_y - 1) * quad_order,
            (num_nodes_x - 1) * quad_order,
        ))
        boundary = raw['boundary'].reshape((
            (num_nodes_y - 1) * quad_order,
            (num_nodes_x - 1) * quad_order,
        ))
        plot_contour(gpt_mesh, fill, plot_fill=True)
        plot_contour(gpt_mesh, boundary, plot_boundary=True)
    elif 0: # param
        multipliers = raw['multipliers'].reshape((num_nodes_y, num_nodes_x))
        plot_contour(mesh, multipliers, plot_fill=True)
    else: # SIMP
        multipliers = raw['multipliers'].reshape((num_nodes_y-1, num_nodes_x-1))
        plot_contour(mesh, multipliers, plot_fill=True)

    plot_save(save='save/save%03i.png'%counter)

    counter += 1
    filename = 'save/data%03i.pkl' % counter

# import movie
