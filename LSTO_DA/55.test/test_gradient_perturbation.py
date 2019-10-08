# first goal is to verify the boundary sensitivities ...

from openmdao.api import Group, Problem, pyOptSparseDriver, view_model
from openmdao.api import IndepVarComp, ExplicitComponent, ImplicitComponent
import sys
sys.path.append('../Density_OpenLSTO')
sys.path.append('../LevelSet_OpenLSTO')
from pylab import *

import scipy.sparse
import scipy.sparse.linalg

from pyBind import py_FEA, py_Sensitivity
from py_lsmBind import py_LSM

from groups.simp_group import SimpGroup #test

from PerturbGroup import PerturbGroup
print("hello world")


# perturb?
isPerturb = True
pertb = 0.2

# FEM Mesh
nelx = 160
nely = 80

length_x = 160.
length_y = 80.

ls2fe_x = length_x/nelx
ls2fe_y = length_y/nely

num_nodes_x = nelx + 1
num_nodes_y = nely + 1

num_nodes = num_nodes_x * num_nodes_y
num_elems = (num_nodes_x - 1) * (num_nodes_y - 1)

# LSM Mesh
num_dofs = num_nodes_x * num_nodes_y * 2 
num_dofs_w_lambda = num_dofs + num_nodes_y * 2

# nodes_plot = get_mesh(num_nodes_x, num_nodes_y, nelx, nely) # for plotting

# FEA properties
E = 1.
nu = 0.3
f = -1.

fea_solver = py_FEA(lx = length_x, ly = length_y, nelx=nelx, nely=nely, element_order=2)
[node, elem, elem_dof] = fea_solver.get_mesh()

nELEM = elem.shape[0]
nNODE = node.shape[0]
nDOF = nNODE * 2

fea_solver.set_material(E=E,nu=nu,rho=1.0)
## BCs ===================================

coord = np.array([0,0])
tol = np.array([1e-3,1e10])
fea_solver.set_boundary(coord = coord,tol = tol)
BCid = fea_solver.get_boundary()

nDOF_withLag  = nDOF + len(BCid)


coord = np.array([length_x,length_y/2])
tol = np.array([1e-3, 1e-3]) 
GF_ = fea_solver.set_force(coord = coord,tol = tol, direction = 1, f = -1.0)
GF = np.zeros(nDOF_withLag)
GF[:nDOF] = GF_

# =========================================

Order_gpts = 2
num_gpts = num_elems * Order_gpts**2


# LSM properties
radius = 2
movelimit = 0.5

# LSM initialize (swisscheese config)

lsm_solver = py_LSM(nelx = nelx, nely = nely, moveLimit = movelimit)
if ((nelx == 160) and (nely == 80)): # 160 x 80 case
    hole = array([[16, 14, 5],
                    [48, 14, 5],
                    [80, 14, 5],
                    [112, 14, 5],
                    [144, 14, 5],

                    [32, 27, 5],
                    [64, 27, 5],
                    [96, 27, 5],
                    [128, 27, 5],

                    [16, 40, 5],
                    [48, 40, 5],
                    [80, 40, 5],
                    [112, 40, 5],
                    [144, 40, 5],

                    [32, 53, 5],
                    [64, 53, 5],
                    [96, 53, 5],
                    [128, 53, 5],

                    [16, 66, 5],
                    [48, 66, 5],
                    [80, 66, 5],
                    [112, 66, 5],
                    [144, 66, 5]],dtype=float)
    if (isPerturb):
        hole = append(hole,[[0., 0., 0.1], [0., 80., 0.1], [160., 0., 0.1], [160., 80., 0.1]], axis = 0)
    lsm_solver.add_holes(locx = list(hole[:,0]), locy = list(hole[:,1]), radius = list(hole[:,2]))


lsm_solver.set_levelset()

# (bpts_xy, areafraction, seglength) = lsm_solver.discretise()
# (rows, cols, vals) = fea_solver.compute_K_SIMP(areafraction)
# nprows = np.array(rows, dtype=np.int32)
# npcols = np.array(cols, dtype=np.int32)
# npvals = np.array(vals, dtype=float)
# K_sparse = scipy.sparse.csc_matrix((npvals, (nprows,npcols))    , 
#                         shape=(num_dofs_w_lambda,num_dofs_w_lambda))
# u = scipy.sparse.linalg.spsolve(K_sparse, GF)[:num_dofs]

# print(min(u)) #= -52.90553842743925
# print(sum(u)) #= -261872.4412214001
# print(np.linalg.norm(u)) #= 2993.8784015467977
# exit(0)

# WIP: testing gradient computationprob.compute_totals()

if 1:
    model = PerturbGroup(
        fea_solver = fea_solver,
        lsm_solver = lsm_solver, 
        nelx = nelx, 
        nely = nely,
        force = GF)
else:
    model = SimpGroup(fam_solver=fea_solver, force=GF,
                      num_elem_x=nelx, num_elem_y=nely,
                      penal=3, volume_fraction=0.5)

prob = Problem(model)
prob.driver = pyOptSparseDriver()
prob.driver.options['optimizer'] = 'IPOPT'
prob.driver.opt_settings['linear_solver'] = 'ma27'

prob.setup(check=False)
prob.run_once()

# rho = prob.get_val('states_comp.multipliers') # no problem (checked)

# u = prob.get_val('disp_comp.disp') # no problem. checked
# print(min(u)) #= -52.90553842743925
# print(sum(u)) #= -261872.4412214001
# print(np.linalg.norm(u)) #= 2993.8784015467977

# exit(0)

f = prob.get_val('compliance_comp.compliance')
print(sum(f))

total = prob.compute_totals() # evoke solve_linear() once.
S_f = total['compliance_comp.compliance','inputs_comp.bpts']
S_g = total['weight_comp.weight','inputs_comp.bpts']

nBpts = int(S_f.shape[1]/2)
S_f = -S_f[0][:nBpts]
S_g = -S_g[0][:nBpts]


# S_f = prob.compute_totals(of=['compliance_comp.compliance'], wrt=['inputs_comp.bpts']) # it puts out zero... 
# S_g = prob.compute_totals(of=['weight_comp.weight'],wrt=['inputs_comp.bpts'])

(bpts_xy, area, lengSeg) = lsm_solver.discretise()

# savetxt("delArea.txt", (bpts_xy, del_area))
savetxt("bpts.txt",bpts_xy)
savetxt("Sens_fg.txt", (S_f,S_g*nelx*nely))


# VERIFIED. FIN.