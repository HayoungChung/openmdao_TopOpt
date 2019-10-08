import numpy as np

from openmdao.api import Problem, view_model, ScipyOptimizer #, pyOptSparseDriver

from fem2d.fem2d import PyFEMSolver
from fem2d.openmdao.fem2d_simp_elemwise_wFilter_group2 import FEM2DSimpGroup
# from fem2d.openmdao.fem2d_simp_elemwise_group import FEM2DSimpGroup
from fem2d.openmdao.fem2d_param_group import FEM2DParamGroup
from fem2d.utils.plot import get_mesh, plot_solution, plot_contour
from fem2d.utils.forces import get_forces


num_nodes_x = 40
num_nodes_y = 20

num_param_x = 41
num_param_y = 21

if 0:
    num_nodes_x = num_param_x = 10; num_nodes_y = num_param_y = 5

length_x = 160
length_y = 80

E = 1.
nu = 0.3
f = -1.
p = 3
w = 0.
quad_order = 5
volume_fraction = 0.4

fem_solver = PyFEMSolver(num_nodes_x, num_nodes_y, length_x, length_y, E, nu)

forces = get_forces(num_nodes_x, num_nodes_y, f=f)
nodes = get_mesh(num_nodes_x, num_nodes_y, length_x, length_y)

if 0:
    model = FEM2DSimpGroup(
        fem_solver=fem_solver,
        length_x=length_x, length_y=length_y,
        num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y,
        forces=forces, p=p,
        nodes=nodes, w=w,
        volume_fraction=volume_fraction)
else:
    model = FEM2DParamGroup(
        fem_solver=fem_solver,
        length_x=length_x, length_y=length_y,
        num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y,
        num_param_x=num_param_x, num_param_y=num_param_y,
        forces=forces, p=p,
        nodes=nodes, w=w, quad_order=quad_order,
        volume_fraction=volume_fraction)

prob = Problem(model)

prob.driver = ScipyOptimizer()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-9
prob.driver.options['disp'] = True

# prob.driver = pyOptSparseDriver()
# prob.driver.options['optimizer'] = 'SNOPT'
# prob.driver.opt_settings['Major optimality tolerance'] = 1e-6
# prob.driver.opt_settings['Major feasibility tolerance'] = 1e-7
# prob.driver.opt_settings['Verify level'] = -1

prob.setup()

# view_model(prob)
# exit()  
if 1:
    prob.run_driver()
else:
    prob.run_model()
    prob.check_partials(compact_print=True)

# disp = prob['disp_comp.disp']
# densities = prob['penalization_comp.y'].reshape((num_nodes_x - 1, num_nodes_y - 1))
#
# nodal_densities = np.zeros((num_nodes_x, num_nodes_y))
# nodal_densities[:-1, :-1] += densities
# nodal_densities[ 1:, :-1] += densities
# nodal_densities[:-1,  1:] += densities
# nodal_densities[ 1:,  1:] += densities
# nodal_densities[1:-1, 1:-1] /= 4.
# nodal_densities[1:-1,  0] /= 2.
# nodal_densities[1:-1, -1] /= 2.
# nodal_densities[ 0, 1:-1] /= 2.
# nodal_densities[-1, 1:-1] /= 2.
#
# scale = 1e0
# disp2 = disp.reshape((num_nodes_x, num_nodes_y, 2))[:, :, 1] * scale
# # plot_solution(orig_nodes, deflected_nodes=deflected_nodes)
# # plot_contour(nodes, field=nodal_densities)
# plot_contour(nodes, densities)

import make_plots
