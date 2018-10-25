# this is a runing script that shows 1-1 comparison
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from openmdao.api import Problem, view_model, ScipyOptimizer
from utils_new.rescale import rescale_lambdas

import sys
sys.path.insert(0, r'/home/hac210/Dropbox/packages/02.M2DO_opensource_new/OpenLSTO/M2DO_FEA/Python/')
# sys.path.insert(0, r'/home/hac210/Dropbox/packages/topOpt_MDO/lib/lsm2d_classwise')
sys.path.insert(0, r'/home/hac210/Dropbox/packages/02.M2DO_opensource_new/OpenLSTO/M2DO_LSM/Python/')
# sys.path.insert(0, r'/home/totoro/Dropbox/packages/02.M2DO_opensource_new/OpenLSTO/M2DO_FEA/Python/')
# sys.path.insert(0, r'/home/totoro/Dropbox/packages/topOpt_MDO/lib/lsm2d_classwise')
# sys.path.insert(0, r'/home/totoro/Dropbox/packages/02.M2DO_opensource_new/OpenLSTO/M2DO_LSM/Python/')
# from fem2d import PyFEMSolver
from pyBind import py_FEA, py_Sensitivity
from py_lsmBind import py_LSM
from utils.forces import get_forces

from groups.lsm2d_SLP_Group_openlsto import LSM2D_slpGroup 
# from lsm_classes import PyLSMSolver

from utils.ls_sensitivity import _LeastSquare

import matplotlib.pyplot as plt
import scipy.optimize as sp_optim

# from utils.optim_refact_v3 import Cantilever

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

# nDOF_withLag  = nDOF + len(BCid)


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

# lsm_solver = PyLSMSolver(num_nodes_x, num_nodes_y, 0.4)

lsm_solver_open = py_LSM(nelx = nelx, nely = nely, moveLimit = movelimit)
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
    lsm_solver_open.add_holes(locx = list(hole[:,0]), locy = list(hole[:,1]), radius = list(hole[:,2]))
# else:
#     lsm_solver_open.add_holes(locx = [], locy = [], radius = [])

lsm_solver_open.set_levelset()



# HJ loop
max_loop = 700
for i_HJ in range(0, max_loop):
    # 0. discretize
    # (bpts_xy0, areafraction0, segmentLength0) = lsm_solver.discretize()
    (bpts_xy, areafraction, seglength) = lsm_solver_open.discretise()
    num_bpts = bpts_xy.shape[0]
    # sys.exit(0)

    if i_HJ == 0:
        bpts_xy0 = bpts_xy
        areafraction0 = areafraction

    (rows, cols, vals) = fem_solver.compute_K_SIMP(areafraction)
    u = fem_solver.solve_FE()

    Order_gpts = 2
    num_gpts = num_elems * Order_gpts**2

    xpos = np.ones(num_elems * 4)
    ypos = np.ones(num_elems * 4)
    sens_compl = np.ones(num_elems * 4)

    pySens = py_Sensitivity(fem_solver, u)
    py_GptSens = pySens.compute_compliance_sens()

    py_bptSens = pySens.compute_boundary_sens(bpts_xy)

    lambdas = np.zeros(2)
    
    # lambdas = lsm_solver.preprocess(lambdas, movelimit, -py_bptSens[:,2])

    # (ub, lb) = lsm_solver.get_bounds()

    # new 
    bpts_sens_new = np.zeros((num_bpts,2))
    bpts_sens_new[:,0] = -py_bptSens[:,2]
    bpts_sens_new[:,1] = -1.0

    # print bpts_sens_new[:,1]
    
    lsm_solver_open.set_BptsSens(bpts_sens_new)
    scales = lsm_solver_open.get_scale_factors()
    (lb2,ub2) = lsm_solver_open.get_Lambda_Limits()
    (lb_,ub_) = lsm_solver_open._get_Lambda_Limits()

    # print (ub,lb)
    # print (ub2, lb2)
    # print scales 

    ## OK up to here: FEB16
    
    # lambdas_tmp = np.random.rand(2)

    # print (lambdas_tmp)

    # displacement0 = lsm_solver.computeDisplacements(lambdas_tmp)
    # displacement_new = lsm_solver_open.compute_displacement(lambdas_tmp)

    # disp_np =  np.array(displacement0)
    # print lsm_solver.computeFunction(disp_np,0)   
    # print lsm_solver.computeFunction(disp_np,1)


    constraint_distance = (0.4 * nelx * nely) - areafraction.sum()
    print areafraction.sum()/(nelx*nely)
    constraintDistance = np.array([(0.4 * nelx * nely) - areafraction.sum()])
    scaled_constraintDist = lsm_solver_open.compute_scaledConstraintDistance(constraintDistance)

    # print "=========================================="
    # print lsm_solver_open.compute_delF(displacement_new)   
    # print lsm_solver_open.compute_delG(displacement_new, scaled_constraintDist, 1)


    # plt.figure(3) 
    # plt.plot(displacement0,'ro')
    # plt.plot(displacement_new,'b*')
    # plt.show()

    # sys.exit()

    if 0:
        def objF_nocallback(x):
            displacement = lsm_solver_open.compute_displacement(x)
            # print('objF')
            # print(displacement)
            displacement_np = np.asarray(displacement)
            # print 'F'
            # print lsm_solver_open.compute_delF(displacement_np)
            return lsm_solver_open.compute_delF(displacement_np)

        def conF_nocallback(x):
            displacement = lsm_solver_open.compute_displacement(x)
            # print('conF')
            # print(displacement)
            displacement_np = np.asarray(displacement)
            # print 'G'
            # print lsm_solver_open.compute_delG(displacement_np, scaled_constraintDist, 1)
            return lsm_solver_open.compute_delG(displacement_np, scaled_constraintDist, 1)

        cons = ({'type': 'eq', 'fun': lambda x: conF_nocallback(x)})
        res = sp_optim.minimize(objF_nocallback, np.zeros(2), method='SLSQP', options={'disp': True},
                                bounds=((lb2[0], ub2[0]), (lb2[1], ub2[1])),
                                constraints=cons)

        lambdas = res.x
        # print(lambdas)
        displacements_ = lsm_solver_open.compute_unscaledDisplacement(lambdas)
        # lambdas = rescale_lambdas(lambdas, displacements_, movelimit)
        # displacements1_ = lsm_solver_open.compute_displacement(lambdas)
        timestep =  abs(lambdas[0]*scales[0])
        
        Bpt_Vel = displacements_ / timestep
    # if 0:
    #     def objF_nocallback(x):
    #         displacement = lsm_solver.computeDisplacements(x)
    #         # print('objF')
    #         # print(displacement)
    #         displacement_np = np.asarray(displacement)
    #         return lsm_solver.computeFunction(displacement_np, 0)[0]

    #     def conF_nocallback(x):
    #         displacement = lsm_solver.computeDisplacements(x)
    #         # print('conF')
    #         # print(displacement)
    #         displacement_np = np.asarray(displacement)
    #         return lsm_solver.computeFunction(displacement_np, 1)[1]
    else:
        # model = LSM2D_slpGroup(
        #     lsm_solver=lsm_solver,
        #     bpts_xy=bpts_xy,  # boundary points
        #     lb=lb, ub=ub,
        # )
        # print lsm_solver_open.compute_displacement(np.zeros(2))

        model = LSM2D_slpGroup(lsm_solver = lsm_solver_open, num_bpts = num_bpts, ub = ub2, lb = lb2,
            Sf = bpts_sens_new[:,0], Sg = bpts_sens_new[:,1], constraintDistance = constraint_distance)
        # model.approx_total_derivs(method = 'fd')
        
        prob = Problem(model)
        prob.setup()
        
        # prob.check_partials(compact_print=True)

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['disp'] = True
        prob.driver.options['tol'] = 1e-6

        prob.run_driver()

        lambdas = prob['inputs_comp.lambdas']
        # print (lambdas)
        
        displacements_ = prob['displacement_comp.displacements']
        
        # lambdas = rescale_lambdas(lambdas, displacements_, movelimit)
        # displacements_ = lsm_solver_open.compute_unscaledDisplacement(lambdas)

        timestep =  abs(lambdas[0]*scales[0])
        
        Bpt_Vel = displacements_ / timestep

    

    # phi0 = lsm_solver.get_phi()
    # lambdas = lsm_solver.postprocess(lambdas)

    # unscaledDisplacement = lsm_solver_open.compute_displacement(lambdas)
    # pro
    
    # timestep = abs(lambdas[0]) # * scales[0])
    # print (unscaledDisplacement, timestep)

    # Bpt_Vel = unscaledDisplacement / timestep

    # lsm_solver.computeVelocities()
    # print(lambdas)
    print ('starts updating')
    plt.figure(1)
    plt.clf()
    plt.scatter(bpts_xy[:,0],bpts_xy[:,1], 30, Bpt_Vel)
    # plt.grid()
    plt.axis("equal")
    plt.colorbar()
    # plt.show()
    plt.savefig("mdo_bpts_%d.png" % i_HJ)

    lsm_solver_open.advect(Bpt_Vel, timestep)
    lsm_solver_open.reinitialise()
    
    # phi = lsm_solver.update(np.abs(lambdas[0]))
    # plt.clf()
    # plt.plot(phi, 'o')
    # plt.savefig('plots2/phi_%d.png' % i_HJ)

    # lsm_solver.reinitialise()

    print ('loop %d is finished' % i_HJ)
    
    # if i_HJ == 0:
    #     raw = {}
    #     raw['mesh'] = nodes
    #     # raw['phi'] = phi0
    #     filename = 'const.pkl'
    #     with open(filename, 'wb') as f:
    #         pickle.dump(raw, f)

    # raw = {}
    # raw['phi'] = phi
    # filename = 'save/data%03i.pkl' % i_HJ
    # with open(filename, 'wb') as f:
    #     pickle.dump(raw, f)

# import make_plots

    # totals = prob.compute_total_derivs(['objective_comp.objective'], ['inputs_comp.lambdas'])
