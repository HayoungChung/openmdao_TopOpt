# this is a up-to-date runing script (update on plotings and savings)
# OCT 18
import numpy as np
from pylab import *
import scipy.sparse
import scipy.sparse.linalg
from openmdao.api import Problem, view_model, ScipyOptimizer, pyOptSparseDriver
from utils_new.rescale import rescale_lambdas

import sys

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

nodes = get_mesh(num_nodes_x, num_nodes_y, nelx, nely) # for plotting

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
if ((nelx == 160) and (nely == 80)): # 160 x 80 case
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
    hole = append(hole,[[0., 0., 0.1], [0., 80., 0.1], [160., 0., 0.1], [160., 80., 0.1]], axis = 0)

    lsm_solver.add_holes(locx = list(hole[:,0]), locy = list(hole[:,1]), radius = list(hole[:,2]))

elif ((nelx == 80) and (nely == 40)): # 160 x 80 case
    hole = np.array([[8, 7, 2.5],
                    [24, 7, 2.5],
                    [40, 7, 2.5],
                    [56, 7, 2.5],
                    [72, 7, 2.5],

                    [16, 13.5, 2.5],
                    [32, 13.5, 2.5],
                    [48, 13.5, 2.5],
                    [64, 13.5, 2.5],

                    [8, 20, 2.5],
                    [24, 20, 2.5],
                    [40, 20, 2.5],
                    [56, 20, 2.5],
                    [72, 20, 2.5],

                    [16, 26.5, 2.5],
                    [32, 26.5, 2.5],
                    [48, 26.5, 2.5],
                    [64, 26.5, 2.5],

                    [8, 33, 2.5],
                    [24, 33, 2.5],
                    [40, 33, 2.5],
                    [56, 33, 2.5],
                    [72, 33, 2.5]],dtype=np.float)
    lsm_solver.add_holes(locx = list(hole[:,0]), locy = list(hole[:,1]), radius = list(hole[:,2]))
# else:
    # lsm_solver.add_holes(locx = [], locy = [], radius = [])

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
    K_sparse = scipy.sparse.csc_matrix((npvals, (nprows,npcols))    , 
                            shape=(num_dofs_w_lambda,num_dofs_w_lambda))
    u = scipy.sparse.linalg.spsolve(K_sparse, GF)[:num_dofs]

    print(min(u))
    print(sum(u))
    print(np.linalg.norm(u))
    exit(0)

    Order_gpts = 2 # number of Gauss Points
    num_gpts = num_elems * Order_gpts**2

    pySens = py_Sensitivity(fem_solver, u)
    py_GptSens = pySens.compute_compliance_sens() # Sensitivities at Gauss points
    py_bptSens = pySens.compute_boundary_sens(bpts_xy) # Sensitivites at Boundary points (using least square)

    # if 0: 
    #     placeholder = np.zeros([num_gpts,3])
    #     for i in range(0, num_gpts):
    #         placeholder[i,0] = py_GptSens[i][0]
    #         placeholder[i,1] = py_GptSens[i][1]
    #         placeholder[i,2] = py_GptSens[i][2]
    #     from pylab import *
    #     figure(3)
    #     clf()
    #     scatter(placeholder[:,0],placeholder[:,1],c= placeholder[:,2])
    #     axis("equal")
    #     # plt.colorbar()
    #     savefig("mdo_gptSens_%d.png" % i_HJ)

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
        
        prob = Problem(model)
        prob.setup()
        
        if 0:
            prob.driver = ScipyOptimizer()
            prob.driver.options['optimizer'] = 'SLSQP'
            prob.driver.options['disp'] = True
            prob.driver.options['tol'] = 1e-10
        else:
            prob.driver = pyOptSparseDriver()
            prob.driver.options['optimizer'] = 'IPOPT'
            prob.driver.opt_settings['linear_solver'] = 'ma27'

        prob.run_driver()
        lambdas = prob['inputs_comp.lambdas']
        displacements_ = prob['displacement_comp.displacements']

        timestep =  abs(lambdas[0]*scales[0])
        Bpt_Vel = displacements_ / timestep

    # advection
    lsm_solver.advect(Bpt_Vel, timestep)
    lsm_solver.reinitialise()
    
    if 1: # quick plot
        plt.figure(1)
        plt.clf()
        plt.scatter(bpts_xy[:,0],bpts_xy[:,1], 30)
        plt.axis("equal")
        plt.savefig("mdo_bpts_%d.png" % i_HJ)
    

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
