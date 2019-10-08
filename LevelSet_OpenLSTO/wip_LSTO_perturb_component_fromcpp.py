# this is a up-to-date runing script (update on plotings and savings)
# try perturbation the boundary
import numpy as np
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
tol = np.array([0.1,0.1]) 
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
    if (isPerturb):
        hole = np.append(hole,[[0., 0., 0.1], [0., 80., 0.1], [160., 0., 0.1], [160., 80., 0.1]], axis = 0)
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
    K_sparse = scipy.sparse.csc_matrix((npvals, (nprows,npcols))    , 
                            shape=(num_dofs_w_lambda,num_dofs_w_lambda))
    u = scipy.sparse.linalg.spsolve(K_sparse, GF)[:num_dofs]

    Order_gpts = 2 # number of Gauss Points
    num_gpts = num_elems * Order_gpts**2

    pySens = py_Sensitivity(fem_solver, u)
    py_GptSens = pySens.compute_compliance_sens() # Sensitivities at Gauss points

    perturb_points_sensitivities = [None] * num_bpts
    perturb_points_indices = [None] * num_bpts
    perturb_boundary_sensitivities = np.zeros((num_bpts,2))

    for bbb in range(0, num_bpts): 
        px_ = bpts_xy[bbb,0]
        py_ = bpts_xy[bbb,1]

        nelx_pert_0 = int(max(int(np.floor(px_)) - 1 - 3, 0))
        nelx_pert_1 = int(min(int(np.floor(px_ - 1e-4)) + 2 + 3, int(nelx))) 
        
        nely_pert_0 = int(max(int(np.floor(py_)) - 1 - 3, 0))
        nely_pert_1 = int(min(int(np.floor(py_ - 1e-4)) + 2 + 3, int(nely)))

        # dimensions of perturbed mesh
        nelx_pert = nelx_pert_1 - nelx_pert_0
        nely_pert = nely_pert_1 - nely_pert_0

        # level-set perturbation mesh
        lsm_pert = py_LSM(nelx = nelx_pert, nely = nely_pert, moveLimit = 0.5)
        lsm_pert.add_holes(locx = [], locy = [], radius = [])
        lsm_pert.set_levelset()

        # assign appropriate signed distance values to the perturbed mesh
        phi_org = lsm_solver.get_phi()

        count_pert = 0
        for iy in range(0, nely_pert+1):
            for ix in range(0, nelx_pert+1):
                global_x = nelx_pert_0 + ix
                global_y = nely_pert_0 + iy

                lsm_pert.set_phi(index = count_pert, value = phi_org[(nelx + 1)*global_y + global_x], isReplace = True)
                count_pert += 1 
        
        lsm_pert.reinitialise()

        (bpts_xy_pert0, areafraction_pert0, seglength_pert0) = lsm_pert.discretise()

        timestep_pert = 1.0 # deltaT for perturbation

        # assign perturbation velocity at the boundary point
        bpt_length = 0.0 
        vel_bpts = np.zeros(bpts_xy_pert0.shape[0])

        # lsm_solver.Print_results(0)
        # lsm_pert.Print_results(1)

        for ii in range(0, bpts_xy_pert0.shape[0]):
            tmp_px_ = bpts_xy_pert0[ii,0]
            tmp_py_ = bpts_xy_pert0[ii,1]
            dist_pert = pow(-tmp_px_ + px_ - nelx_pert_0, 2) + pow(-tmp_py_ + py_ - nely_pert_0, 2)
            dist_pert = dist_pert**0.5
            if (dist_pert < pertb):
                vel_bpts[ii] = pertb * (1.0 - pow(dist_pert/pertb, 2.0))
            else:
                vel_bpts[ii] = 0.0
            # print(tmp_px_, tmp_py_, vel_bpts[ii])
        
        lsm_pert.advect_woWENO(vel_bpts, timestep_pert)

        # discretize again to get perturbed data
        (bpts_xy_pert1, areafraction_pert1, seglength_pert1) = lsm_pert.discretise()
        # lsm_pert.Print_results(2)

        # loop through the elements in the narrow band
        perturb_sensitivities = []
        perturb_indices = []
        count_pert = 0
        for iy in range(0, nely_pert):
            for ix in range(0, nelx_pert):
                global_x = nelx_pert_0 + ix
                global_y = nely_pert_0 + iy
                global_index = (nelx)*global_y + global_x

                delta_x = min(areafraction_pert0[count_pert] - areafraction_pert1[count_pert], 0.8*pertb)


                if (delta_x > 1e-3 * pertb):
                    perturb_sensitivities = np.append(perturb_sensitivities, delta_x)
                    perturb_indices = np.append(perturb_indices, global_index)

                count_pert += 1
        
        perturb_points_sensitivities[bbb] = perturb_sensitivities
        perturb_points_indices[bbb] = perturb_indices

    # computes sensitivity based on the perturbation method
    delArea_list = np.zeros(num_bpts)
    for bbb in range(0, bpts_xy.shape[0]):
        # px_ = bpts_xy[bbb,0]
        # py_ = bpts_xy[bbb,1]
        delta_sensi = 0.0
        delta_area = 0.0

        # find approperiate element
        for eee in range(0, len(perturb_points_indices[bbb])):
            cur_index = perturb_points_indices[bbb][eee]
            delta_area += perturb_points_sensitivities[bbb][eee]

            for ggg in range(0, 4):
                delta_sensi += pySens.get_elem_gpts_sens(cur_index,ggg) * perturb_points_sensitivities[bbb][eee] / 4.0
                # print(pySens.get_elem_gpts_sens(cur_index,ggg))
            
            # if (eee == 2):
            #     exit()
    
        delArea_list[bbb] = delta_area # verified

        perturb_boundary_sensitivities[bbb,0] = -delta_sensi / seglength[bbb] / pertb # error...
        # make sure area sensitivity is between -1.5 and -0.5
        perturb_boundary_sensitivities[bbb,1] = -min(delta_area / pertb / seglength[bbb], 1.5)
        perturb_boundary_sensitivities[bbb,1] = min(perturb_boundary_sensitivities[bbb,1], -0.5) #verified

    py_bptSens = pySens.compute_boundary_sens(bpts_xy) # Sensitivites at Boundary points (using least square)
    
    if 0: # comparison of sensitivity w.r.t. cpp version
        a = np.loadtxt("../LSTO_perturbation/cpp_1.txt")
        gsens_cpp  =  np.loadtxt("gpts_Sens.txt")

        plt.figure(0) # boudnary sensitivity
        plt.subplot(3,1,1)
        plt.scatter(a[:,0], a[:,1], s =10, c = a[:,2])
        plt.colorbar()

        plt.figure(2) # area
        plt.subplot(2,1,1)
        plt.scatter(a[:,0], a[:,1], s =10, c = a[:,3])
        plt.colorbar()

        plt.figure(15) # delta_sensi```
        plt.scatter(a[:,0], a[:,1], s =10, c = a[:,4])
        plt.colorbar()

        plt.figure(2) # area 
        plt.subplot(2,1,2)
        plt.scatter(bpts_xy[:,0], bpts_xy[:,1], s =10, c = delArea_list)
        plt.colorbar()

        plt.figure(1) # (correct) Gpts_sens
        plt.subplot(3,1,1)
        plt.scatter(py_GptSens[:,0], py_GptSens[:,1], s =5, c = py_GptSens[:,2])
        plt.colorbar()
        plt.subplot(3,1,2)
        plt.scatter(gsens_cpp[:,0], gsens_cpp[:,1], s =5, c = gsens_cpp[:,2])
        plt.colorbar()
        plt.subplot(3,1,3)
        plt.plot(gsens_cpp[:,2], py_GptSens[:,2],'o')
        plt.plot(py_GptSens[:,2], py_GptSens[:,2],'-')


        plt.figure(0) # bpts_sens
        plt.subplot(3,1,2)
        plt.scatter(bpts_xy[:,0], bpts_xy[:,1], s =10, c = -py_bptSens[:,2])
        plt.colorbar()
        
        plt.figure(0) # perturbed_sens
        plt.subplot(3,1,3)
        plt.scatter(bpts_xy[:,0], bpts_xy[:,1], s =10, c = perturb_boundary_sensitivities[:,0])
        plt.colorbar()

        plt.figure(11)
        plt.subplot(2,1,1) #FIXME: error in a low-sensitivity region
        plt.plot(perturb_boundary_sensitivities[:,0], a[:,2], 'o')
        plt.plot(a[:,2], a[:,2], '-',)
        plt.grid()
        plt.subplot(2,1,2)
        plt.plot(perturb_boundary_sensitivities[:,1], a[:,5], 'o')
        plt.plot(a[:,5], a[:,5], '-',)
        plt.grid()

        plt.figure(12) #area
        plt.plot(delArea_list, a[:,3], 'o')
        plt.plot(a[:,3], a[:,3], '-',)
        plt.grid()


        plt.show()
 
    if 0: 
        plt.figure(0) # perturbed_sens
        plt.clf()
        plt.subplot(3,1,1)
        plt.scatter(bpts_xy[:,0], bpts_xy[:,1], s =1, c = perturb_boundary_sensitivities[:,0])
        plt.colorbar()
        plt.axis("equal")


        plt.subplot(3,1,2)
        plt.scatter(bpts_xy[:,0], bpts_xy[:,1], s =1, c = delArea_list)
        plt.colorbar()
        plt.axis("equal")
        plt.subplot(3,1,3) # symmetry check
        plt.scatter(bpts_xy[:,0], 80-bpts_xy[:,1], s =1, c = delArea_list)
        plt.colorbar()
        plt.axis("equal")


        plt.savefig("mdo_sens_%d.png" % i_HJ)

    if 0: 
        a = np.loadtxt("../LSTO_perturbation/cpp_" + str(1) + ".txt")
        plt.figure(0) # perturbed_sens
        plt.clf()
        plt.subplot(3,1,1)
        plt.scatter(bpts_xy[:,0], bpts_xy[:,1], s =1, c = perturb_boundary_sensitivities[:,0])
        plt.colorbar()
        plt.axis("equal")


        plt.subplot(3,1,2)
        plt.scatter(bpts_xy[:,0], bpts_xy[:,1], s =1, c = -py_bptSens[:,2])
        plt.colorbar()
        plt.axis("equal")


        plt.subplot(3,1,3)
        plt.scatter(bpts_xy[:,0], bpts_xy[:,1], s =1, c = a[:,2])
        plt.colorbar()
        plt.axis("equal")


        print([min(-py_bptSens[:,2]), min(a[:,2]), min(perturb_boundary_sensitivities[:,0])])
        print([max(-py_bptSens[:,2]), max(a[:,2]), max(perturb_boundary_sensitivities[:,0])])

        plt.show()
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
    # bpts_sens_new[:,0] = -py_bptSens[:,2]
    # bpts_sens_new[:,1] = -1.0 # WIP: is there any asymmetry? if not, area computation is where ther error comes from...

    # bpts_sens_new[:,0] = perturb_boundary_sensitivities[:,0]
    # bpts_sens_new[:,1] = perturb_boundary_sensitivities[:,1]

    a = np.loadtxt("../LSTO_perturbation/cpp_" + str(i_HJ+1) + ".txt")

    bpts_sens_new[:,0] = a[:,2]
    bpts_sens_new[:,1] = a[:,5]

    lsm_solver.set_BptsSens(bpts_sens_new)
    scales = lsm_solver.get_scale_factors()
    (lb2,ub2) = lsm_solver.get_Lambda_Limits()

    constraint_distance = (0.5 * nelx * nely) - areafraction.sum()
    
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
        
        if 1: # FIXME: not much difference
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

        timestep = abs(lambdas[0]*scales[0])
        Bpt_Vel = displacements_ / timestep

    # advection
    lsm_solver.advect(Bpt_Vel, timestep)
    lsm_solver.reinitialise()
    
    if 1: # quick plot
        plt.figure(1)
        plt.clf()
        (bpts_xy_, areafraction_, seglength_) = lsm_solver.discretise()

        plt.scatter(bpts_xy_[:,0],bpts_xy_[:,1], 30)
        plt.axis("equal")
        plt.savefig("Sensfromcpp/mdo_bpts_%d.png" % i_HJ)
    

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
