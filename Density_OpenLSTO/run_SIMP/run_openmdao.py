import numpy as np
import scipy.sparse.linalg

from openmdao.api import Problem, view_model, ScipyOptimizer  # , pyOptSparseDriver

from pyBind import py_FEA, py_Sensitivity
from simp_w_filter_group import SimpGroup
# from fem2d.openmdao.fem2d_simp_elemwise_wFilter_group2 import FEM2DSimpGroup
# from fem2d.openmdao.fem2d_param_group import FEM2DParamGroup
# from fem2d.utils.plot import get_mesh, plot_solution, plot_contour
# from fem2d.utils.forces import get_forces

# Meshing =============================
nelx = 40#160
nely = 20#80

fem_solver = py_FEA(nelx=nelx, nely=nely, element_order=2)
[node, elem] = fem_solver.get_mesh()

nELEM = elem.shape[0]
nNODE = node.shape[0]
nDOF = nNODE * 2

# Material =============================
E = 1.0
nu = 0.3
fem_solver.set_material(E=E, nu=nu, rho=1)

# Boundary condition  ==================
coord = np.array([0, 0])
tol = np.array([1e-3, 1e10])
fem_solver.set_boundary(coord=coord, tol=tol)
BCid = np.array(fem_solver.get_boundary())

# force in np ==========================
coord = np.array([nelx, nely / 2])
tol = np.array([1e-3, 1e-3])
f = -1.
GF = np.array(fem_solver.set_force(coord = coord,tol = tol, direction = 1, f = f),dtype=float)

# ======================================
p = 3
w = 0.
quad_order = 5
volume_fraction = 0.4

# # Obtain K matrix =======================
# (rows, cols, vals) = fem_solver.compute_K()
# nprows = np.array(rows)
# npcols = np.array(cols)
# npvals = np.array(vals)

# mtx = scipy.sparse.csc_matrix((npvals, (nprows, npcols)), shape=(nDOF, nDOF))
# state = scipy.sparse.linalg.spsolve(mtx, GF)  # direct solver is used


if 1:
    model = SimpGroup(fem_solver=fem_solver, force=GF, BCid = BCid,
                      num_elem_x=nelx, num_elem_y=nely,
                      penal=p, volume_fraction=volume_fraction)
else:
    model = ParamGroup(fem_solver=fem_solver, force=GF,
                       num_elem_x=nelx, num_elem_y=nely,
                       num_param_x=num_param_x, num_param_y=num_param_y,
                       penal=p, volume_fraction=volume_fraction)


# optimizer setup
prob = Problem(model)
prob.driver = ScipyOptimizer()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-9
prob.driver.options['disp'] = True

prob.setup()
# view_model(prob)

if 0:
    prob.run_driver()
else:
    prob.run_model()
    prob.check_partials(compact_print=True)

# disp = prob['states_comp.states']
# density = prob['inputs_comp.dvs']

Q = prob.compute_total_derivs()

np.savetxt("obj_dvs.txt",np.array(Q['objective_comp.objective', 'inputs_comp.dvs']))
np.savetxt("wght_dvs.txt",np.array(Q['weight_comp.weight', 'inputs_comp.dvs']))


# W = prob.check_total_derivatives()
# np.savetxt("obj_dvs_approx.txt",W['objective_comp.objective', 'inputs_comp.dvs'])
# np.savetxt("wght_dvs_approx.txt",W['weight_comp.weight', 'inputs_comp.dvs'])


# Q = prob.compute_total_derivs("objective_comp.objective","inputs_comp.dvs")
# W = prob.compute_total_derivs(prob['objective_comp.weight'],prob['inputs_comp.dvs'])

# np.savetxt("obj_dvs.txt",Q)
# np.savetxt("weight_dvs.txt",W)





# import make_plots
