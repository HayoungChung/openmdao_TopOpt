# this is a up-to-date runing script (update on plotings and savings)
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from openmdao.api import Problem, view_model, ScipyOptimizer
from utils_new.rescale import rescale_lambdas

import sys
# sys.path.insert(0, r'/home/hac210/Dropbox/packages/02.M2DO_opensource_new/OpenLSTO/M2DO_FEA/Python/')
# sys.path.insert(0, r'/home/hac210/Dropbox/packages/02.M2DO_opensource_new/OpenLSTO/M2DO_LSM/Python/')
sys.path.insert(0, r'/home/hac210/00.Working/test_OpenMDAO_LSTO/OpenLSTO-master/M2DO_FEA/Python/')
sys.path.insert(0, r'/home/hac210/00.Working/test_OpenMDAO_LSTO/OpenLSTO-master/M2DO_LSM/Python/')

from pyBind import py_FEA, py_Sensitivity
from py_lsmBind import py_LSM
from utils.forces import get_forces

from groups.lsm2d_SLP_Group_openlsto import LSM2D_slpGroup 

from utils.ls_sensitivity import _LeastSquare

import matplotlib.pyplot as plt
import scipy.optimize as sp_optim

from plot import get_mesh, plot_solution, plot_contour

try:
    import cPickle as pickle
except:
    import pickle


# FEM Mesh
num_nodes_x = 161
num_nodes_y = 81
nelx = num_nodes_x - 1
nely = num_nodes_y - 1

num_nodes = num_nodes_x * num_nodes_y
num_elems = (num_nodes_x - 1) * (num_nodes_y - 1)

# LSM Mesh
length_x = num_nodes_x - 1
length_y = num_nodes_y - 1

num_dofs = num_nodes_x * num_nodes_y * 2 
num_dofs_w_lambda = num_dofs + num_nodes_y * 2

nodes = get_mesh(num_nodes_x, num_nodes_y, length_x, length_y)

# FEA properties
E = 1.
nu = 0.3
f = -1.

fem_solver = py_FEA(lx = length_x, ly = length_y, nelx=nelx, nely=nely, element_order=2)
[node, elem, elem_dof] = fem_solver.get_mesh()

## BCs ===================================
fem_solver.set_material(E,nu,1.0)

coord = np.array([0,0])
tol = np.array([1e-3,1e10])
fem_solver.set_boundary(coord = coord,tol = tol)
BCid = fem_solver.get_boundary()

coord = np.array([length_x,length_y/2])
tol = np.array([1,1]) 
GF_ = fem_solver.set_force(coord = coord,tol = tol, direction = 1, f = -1.0)
GF = np.zeros(num_dofs_w_lambda)
GF[:num_dofs] = GF_
# =========================================

Order_gpts = 2
num_gpts = num_elems * Order_gpts**2


# LSM properties
radius = 2
movelimit = 0.5

# LSM initialize (swisscheese config)

lsm_solver = py_LSM(nelx = nelx, nely = nely, moveLimit = movelimit)
if 0:
    hole = np.array([[16, 14, 5],
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
                    [144, 66, 5]],dtype=np.float)
    lsm_solver.add_holes(locx = list(hole[:,0]), locy = list(hole[:,1]), radius = list(hole[:,2]))
# else:
#     lsm_solver.add_holes(locx = [], locy = [], radius = [])

lsm_solver.set_levelset()



# HJ loop
max_loop = 700
for i_HJ in range(0, max_loop):
    # 0. discretize
    (bpts_xy, areafraction, seglength) = lsm_solver.discretise()
    num_bpts = bpts_xy.shape[0]

    if i_HJ == 0:
        bpts_xy0 = bpts_xy
        areafraction0 = areafraction

    (rows, cols, vals) = fem_solver.compute_K_SIMP(areafraction)
    nprows = np.array(rows, dtype=np.int32)
    npcols = np.array(cols, dtype=np.int32)
    npvals = np.array(vals, dtype=float)
    # u = fem_solver.solve_FE()
    K_sparse = scipy.sparse.csc_matrix((npvals, (nprows,npcols))    , 
                            shape=(num_dofs_w_lambda,num_dofs_w_lambda))
    u_1 = scipy.sparse.linalg.spsolve(K_sparse, GF)

    u = u_1[:num_dofs]

    Order_gpts = 2
    num_gpts = num_elems * Order_gpts**2

    pySens = py_Sensitivity(fem_solver, u)
    py_GptSens = pySens.compute_compliance_sens()
    # if 0:
    #     placeholder = np.zeros([num_gpts,3])
    #     for i in range(0, num_gpts):
    #         placeholder[i,0] = py_GptSens[i][0]
    #         placeholder[i,1] = py_GptSens[i][1]
    #         placeholder[i,2] = py_GptSens[i][2]
    #     from pylab import *
    #     scatter(placeholder[:,0],placeholder[:,1],c= placeholder[:,2])
    #     show()

    py_bptSens = pySens.compute_boundary_sens(bpts_xy)
    # print py_bptSens
    # if 1:
    #     placeholder = np.zeros([num_bpts,3])
    #     for i in range(0, num_bpts):
    #         placeholder[i,0] = py_bptSens[i][0]
    #         placeholder[i,1] = py_bptSens[i][1]
    #         placeholder[i,2] = py_bptSens[i][2]
    #     from pylab import *
    #     scatter(placeholder[:,0],placeholder[:,1],c= placeholder[:,2])
    #     show()

    lambdas = np.zeros(2)
    
    bpts_sens_new = np.zeros((num_bpts,2))
    bpts_sens_new[:,0] = -py_bptSens[:,2]
    bpts_sens_new[:,1] = -1.0
    
    lsm_solver.set_BptsSens(bpts_sens_new)
    scales = lsm_solver.get_scale_factors()
    (lb2,ub2) = lsm_solver.get_Lambda_Limits()

    constraint_distance = (0.4 * nelx * nely) - areafraction.sum()
    
    if 0: # it works for now. (10/24)
        constraintDistance = np.array([constraint_distance])
        scaled_constraintDist = lsm_solver.compute_scaledConstraintDistance(constraintDistance)

        def objF_nocallback(x):
            displacement = lsm_solver.compute_displacement(x)
            displacement_np = np.asarray(displacement)
            return lsm_solver.compute_delF(displacement_np)

        def conF_nocallback(x):
            displacement = lsm_solver.compute_displacement(x)
            displacement_np = np.asarray(displacement)
            return lsm_solver.compute_delG(displacement_np, scaled_constraintDist, 1)

        cons = ({'type': 'eq', 'fun': lambda x: conF_nocallback(x)})
        res = sp_optim.minimize(objF_nocallback, np.zeros(2), method='SLSQP', options={'disp': True},
                                bounds=((lb2[0], ub2[0]), (lb2[1], ub2[1])),
                                constraints=cons)

        lambdas = res.x
        displacements_ = lsm_solver.compute_unscaledDisplacement(lambdas)
        timestep =  abs(lambdas[0]*scales[0])
        
        Bpt_Vel = displacements_ / timestep

    else:
        model = LSM2D_slpGroup(lsm_solver = lsm_solver, num_bpts = num_bpts, ub = ub2, lb = lb2,
            Sf = bpts_sens_new[:,0], Sg = bpts_sens_new[:,1], constraintDistance = constraint_distance)
        # model.approx_total_derivs(method = 'fd')

        
        prob = Problem(model)
        prob.setup()
        
        # view_model(prob)
        # exit(1)
        # prob.check_partials(compact_print=True)

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['disp'] = True
        prob.driver.options['tol'] = 1e-10

        prob.run_driver()
        lambdas = prob['inputs_comp.lambdas']
        displacements_ = prob['displacement_comp.displacements']

        # lambdas = rescale_lambdas(lambdas, displacements_, movelimit)
        # displacements_ = lsm_solver.compute_unscaledDisplacement(lambdas)

        timestep =  abs(lambdas[0]*scales[0])
        Bpt_Vel = displacements_ / timestep

    if 1: # quick plot
        plt.figure(1)
        plt.clf()
        plt.scatter(bpts_xy[:,0],bpts_xy[:,1], 30)
        # plt.show()
        plt.axis("equal")
        # plt.colorbar()
        plt.savefig("mdo_bpts_%d.png" % i_HJ)
    
    # advection
    lsm_solver.advect(Bpt_Vel, timestep)
    lsm_solver.reinitialise()
    
    print ('loop %d is finished' % i_HJ)
    area = areafraction.sum()/(nelx*nely)
    compliance = np.dot(u,GF_)

    print (compliance, area)    

    fid = open("save/log.txt","a+")
    fid.write(str(compliance) + ", " + str(area) + "\n")
    fid.close()

    phi = lsm_solver.get_phi()
    
    if i_HJ == 0:
        raw = {}
        raw['mesh'] = nodes
        filename = 'save/const.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(raw, f)

    raw = {}
    raw['phi'] = phi
    filename = 'save/data%03i.pkl' % i_HJ
    with open(filename, 'wb') as f:
        pickle.dump(raw, f)

# import make_plots
