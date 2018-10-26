import numpy as np
import scipy as sp
import scipy.sparse.linalg
import sys
# sys.path.insert(0, r'/home/hac210/Dropbox/packages/02.M2DO_opensource_new/OpenLSTO/M2DO_FEA/Python/')
# sys.path.insert(0, r'/home/hac210/00.Working/test_OpenMDAO_LSTO/OpenLSTO-master/M2DO_FEA/Python/')

from openmdao.api import Problem, view_model, ScipyOptimizer  # , pyOptSparseDriver

from pyBind import py_FEA, py_Sensitivity
# from groups.simp_w_filter_group import SimpGroup
from groups.simp_group import SimpGroup
from groups.param_group import FEM2DParamGroup as ParamGroup
from utils.plot import get_mesh, plot_solution, plot_contour
# from fem2d.utils.forces import get_forces

# Meshing =============================
nelx = 80
nely = 40

length_x = 160
length_y = 80

num_param_x = int(nelx/2. + 1) #41
num_param_y = int(nely/2. + 1) #21

fem_solver = py_FEA(lx = length_x, ly = length_y, nelx=nelx, nely=nely, element_order=2)
[node, elem, elem_dof] = fem_solver.get_mesh()

nELEM = elem.shape[0]
nNODE = node.shape[0]
nDOF = nNODE * 2

# Material =============================
E = 1.0
nu = 0.3
fem_solver.set_material(E=E, nu=nu, rho=1)

# # Boundary condition  ==================

coord = np.array([0,0])
tol = np.array([1e-3,1e10])
fem_solver.set_boundary(coord = coord,tol = tol)
BCid = fem_solver.get_boundary()

nDOF_withLag  = nDOF + len(BCid)


coord = np.array([length_x,length_y/2])
tol = np.array([1e-3, 1e-3]) 
GF_ = fem_solver.set_force(coord = coord,tol = tol, direction = 1, f = -1.0)
GF = np.zeros(nDOF_withLag)
GF[:nDOF] = GF_

# Quickcheck for FEA
if 0:
    (rows, cols, vals) = fem_solver.compute_K()
    u = fem_solver.solve_FE()
    # print np.min(u)
    # if 160 x 80 : min(u) = -40.201


# ======================================
p = 3
w = 0.
quad_order = 5
volume_fraction = 0.4

# # Obtain K matrix =======================

if 1:
    model = SimpGroup(fem_solver=fem_solver, force=GF,
                      num_elem_x=nelx, num_elem_y=nely,
                      penal=p, volume_fraction=volume_fraction)
else:
    model = ParamGroup(fem_solver=fem_solver, forces=GF,
                    #    num_elem_x=nelx, num_elem_y=nely,
                       length_x=length_x, length_y=length_y,
                       num_nodes_x=nelx+1, num_nodes_y = nely+1,
                       num_param_x=num_param_x, num_param_y=num_param_y,
                       p=p, w = 0., quad_order=quad_order, volume_fraction=volume_fraction)


# optimizer setup
prob = Problem(model)

prob.driver = ScipyOptimizer()
prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['tol'] = 1e-5 
#prob.driver.options['maxiter'] = 
prob.driver.options['disp'] = True

prob.setup()
# view_model(prob)

# if 1:
#     prob.run_driver()
# else:
#     prob.check_partials(compact_print=True)
#     prob.run_model()

prob.run_model()
totals = prob.compute_totals()
print(totals['objective_comp.objective', 'inputs_comp.dvs'][0][0])


# import make_plots
