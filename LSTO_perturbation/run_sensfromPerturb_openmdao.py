# first goal is to verify the boundary sensitivities ...

from openmdao.api import Group, Problem, pyOptSparseDriver, view_model
from openmdao.api import IndepVarComp, ExplicitComponent, ImplicitComponent
import sys
sys.path.append('../Density_OpenLSTO')
sys.path.append('../LevelSet_OpenLSTO')
from plot import get_mesh, plot_solution, plot_contour
try:
    import cPickle as pickle
except:
    import pickle

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

nodes = get_mesh(num_nodes_x, num_nodes_y, nelx, nely) # for plotting

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
    if (isPerturb):
        hole = append(hole,[[0., 0., 0.1], [0., 40., 0.1], [80., 0., 0.1], [80., 40., 0.1]], axis = 0)
    lsm_solver.add_holes(locx = list(hole[:,0]), locy = list(hole[:,1]), radius = list(hole[:,2]))


lsm_solver.set_levelset()

for i_HJ in range(240):
    (bpts_xy, areafraction, seglength) = lsm_solver.discretise()

    if 1:
        model = PerturbGroup(
            fea_solver = fea_solver,
            lsm_solver = lsm_solver, 
            nelx = nelx, 
            nely = nely,
            force = GF, movelimit = movelimit)
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

    total = prob.compute_totals() # evoke solve_linear() once.
    Sf = total['compliance_comp.compliance','inputs_comp.Vn']
    Sg = total['weight_comp.weight','inputs_comp.Vn']

    nBpts = int(bpts_xy.shape[0])
    Sf = -Sf[0][:nBpts]
    Sg = -Sg[0][:nBpts]*length_x*length_y

    # suboptimization
    if 1:  # bisection..
        Cf = np.multiply(Sf, seglength)
        Cg = np.multiply(-Sg, seglength)

        percent_area = 0.5
        target_area = sum(areafraction)

        # target area
        for ii in range(nBpts):
            target_area += Cg[ii] * percent_area * (-movelimit)
        target_area = max(0.5 * length_x * length_y, target_area)

        print("target = ")
        print(target_area/length_x/length_y)

        # distance vector
        domain_distance_vector = np.zeros(nBpts)
        for ii in range(nBpts): 
            px_ = bpts_xy[ii,0]
            py_ = bpts_xy[ii,1]
            # assume square design domain
            domdist = min([abs(px_ -0.0), abs(px_ - length_x), abs(py_ - length_y), abs(py_ - 0.0)])
            if ( (px_ >= length_x) or ( px_ <= 0.0) or (py_ >= length_y) or (py_ <= 0.0) ):
                domdist = -1.0 * domdist
            
            domain_distance_vector[ii] = min(domdist, movelimit)

        lambda_0 = 0.0 # default parameter
        default_area = sum(areafraction)
        for ii in range(nBpts):
            default_area += Cg[ii]*min(domain_distance_vector[ii], movelimit*Sg[ii] + lambda_0*Sf[ii])


        delta_lambda = 0.1 # perturbation
        for iITER in range(20):
            
            lambda_curr = lambda_0
            new_area0 = sum(areafraction)
            for kk in range(nBpts):
                new_area0 += Cg[kk]*min( domain_distance_vector[kk], movelimit*Sg[kk] + lambda_curr*Sf[kk] )

            lambda_curr = lambda_0 + delta_lambda
            new_area2 = sum(areafraction)
            for kk in range(nBpts):
                new_area2 += Cg[kk]*min( domain_distance_vector[kk], movelimit*Sg[kk] + lambda_curr*Sf[kk] )

            lambda_curr = lambda_0 - delta_lambda
            new_area1 = sum(areafraction)
            for kk in range(nBpts):
                new_area1 += Cg[kk]*min( domain_distance_vector[kk], movelimit*Sg[kk] + lambda_curr*Sf[kk] )

            slope = (new_area2 - new_area1) / 2 / delta_lambda

            lambda_0 -= (new_area0 - target_area) / slope

            # termination
            if (abs(new_area0 - target_area) < 1.0E-3):
                print([new_area0/length_x/length_y, target_area/length_x/length_y])
                break
            
        # iteration fin

    lambda_f = lambda_0 
        
    # velocity calculation
    Bpt_Vel = np.zeros(nBpts)
    for ii in range(nBpts):
        domdist = domain_distance_vector[ii]
        Bpt_Vel[ii] = -1.0*min( lambda_f*Sf[ii] + movelimit*Sg[ii], domdist)
    
    abs_Vel = max(np.abs(Bpt_Vel))

    if (abs_Vel > movelimit):
        Bpt_Vel *= movelimit/abs_Vel

    timestep = 1.0   

    lsm_solver.advect(Bpt_Vel, timestep)
    lsm_solver.reinitialise()

    if 1: # quick plot
        plt.figure(1)
        plt.clf()
        plt.scatter(bpts_xy[:,0],bpts_xy[:,1], 10)
        plt.axis("equal")
        plt.savefig("save/figs/mdo_bpts_%d.png" % i_HJ)


    print ('loop %d is finished' % i_HJ)
    area = areafraction.sum()/(nelx*nely)
    u = prob['disp_comp.disp']
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
