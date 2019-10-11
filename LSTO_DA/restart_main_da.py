# restart script
# use Pickle to import level set function
# Before use this script, TODO: always check if folder-name / force / Bcs are consistent with the main_da.

from openmdao.api import Group, Problem, view_model, pyOptSparseDriver
from openmdao.api import IndepVarComp, ExplicitComponent, ImplicitComponent
from post.plot import get_mesh, plot_solution, plot_contour
from matplotlib import pyplot as plt

import cPickle as pickle
import numpy as np
from psutil import virtual_memory

# imports Cython wrappers for OpenLSTO_FEA, OpenLSTO_LSM
from pyBind import py_FEA
from py_lsmBind import py_LSM

# imports perturbation method (aka discrete adjoint)
from groups.PerturbGroup import *

# imports solvers for suboptimization
# TODO: needs to be replaced with OpenMDAO optimizer
from suboptim.solvers import Solvers

loadFolder0 = "./save/"  # NB: must be3 equal to run_main_da.py

def main(*args):

    objectives = {0: "compliance", 1: "stress",
                2: "conduction", 3: "coupled_heat"}

    loadFolder = loadFolder0 + ""
    restart_iter = 47

    for x in args:




    import os
    try:
        os.mkdir(loadFolder + 'restart_' + str(restart_iter))
    except:
        pass

    try:
        os.mkdir(loadFolder + 'restart_' + str(restart_iter) + '/figs')
    except:
        pass


    inspctFlag = False
    if tot_iter < 0:
        inspctFlag = True
        tot_iter = restart_iter + 1

    # select which problem to solve
    obj_flag = 2
    print(locals())
    print("solving %s problem" % objectives[obj_flag])

    print("restarting from %d ..." % restart_iter)
    fname0 = loadFolder + 'phi%03i.pkl' % restart_iter

    with open(fname0, 'rb') as f:
        raw = pickle.load(f)

    phi0 = raw['phi']

    fname0 = loadFolder0 + 'const.pkl'
    with open(fname0, 'rb') as f:
        raw = pickle.load(f)

    # nodes = raw['mesh']
    nodes = raw['nodes']
    elem = raw['elem']
    GF_e = raw['GF_e']
    GF_t = raw['GF_t']
    BCid_e = raw['BCid_e']
    BCid_t = raw['BCid_t']
    E = raw['E']
    nu = raw['nu']
    f = raw['f']
    K_cond = raw['K_cond']
    alpha = raw['alpha']
    nelx = raw['nelx']
    nely = raw['nely']
    length_x = raw['length_x']
    length_y = raw['length_y']
    coord_e = raw['coord_e']
    tol_e = raw['tol_e']

    ########################################################
    ################# 		FEA 		####################
    ########################################################
    # NB: only Q4 elements + integer-spaced mesh are assumed

    ls2fe_x = length_x/float(nelx)
    ls2fe_y = length_y/float(nely)

    num_nodes_x = nelx + 1
    num_nodes_y = nely + 1

    nELEM = nelx * nely
    nNODE = num_nodes_x * num_nodes_y

    # Declare FEA object (OpenLSTO_FEA) ======================
    fea_solver = py_FEA(lx=length_x, ly=length_y,
                        nelx=nelx, nely=nely, element_order=2)
    [node, elem, elem_dof] = fea_solver.get_mesh()

    # validate the mesh
    if nELEM != elem.shape[0]:
        error("error found in the element")
    if nNODE != node.shape[0]:
        error("error found in the node")

    nDOF_t = nNODE * 1  # each node has one temperature DOF
    nDOF_e = nNODE * 2  # each node has two displacement DOFs

    # constitutive properties =================================
    fea_solver.set_material(E=E, nu=nu, rho=1.0)

    # Boundary Conditions =====================================
    fea_solver.set_boundary(coord=coord_e, tol=tol_e)
    BCid_e = fea_solver.get_boundary()
    nDOF_e_wLag = nDOF_e + len(BCid_e)  # elasticity DOF
    nDOF_t_wLag = nDOF_t + len(BCid_t)  # temperature DOF

    ########################################################
    ################# 		LSM 		####################
    ########################################################
    movelimit = 0.5

    # Declare Level-set object
    lsm_solver = py_LSM(nelx=nelx, nely=nely, moveLimit=movelimit)
    lsm_solver.add_holes([], [], [])
    lsm_solver.set_levelset()

    lsm_solver.set_phi_re(phi0)

    lsm_solver.reinitialise()

    for i_HJ in range(restart_iter, tot_iter):

        (bpts_xy, areafraction, seglength) = lsm_solver.discretise()

        ########################################################
        ############### 		OpenMDAO 		################
        ########################################################

        # Declare Group
        if (objectives[obj_flag] == "compliance"):
            model = ComplianceGroup(
                fea_solver=fea_solver,
                lsm_solver=lsm_solver,
                nelx=nelx,
                nely=nely,
                force=GF_e, movelimit=movelimit, BCid = BCid_e)
        elif (objectives[obj_flag] == "stress"):
            # TODO: sensitivity has not been verified yet
            model = StressGroup(
                fea_solver=fea_solver,
                lsm_solver=lsm_solver,
                nelx=nelx,
                nely=nely,
                force=GF_e, movelimit=movelimit,
                pval=5.0, E=E, nu=nu)
        elif (objectives[obj_flag] == "conduction"):
            model = ConductionGroup(
                fea_solver=fea_solver,
                lsm_solver=lsm_solver,
                nelx=nelx,
                nely=nely,
                force=GF_t, movelimit=movelimit,
                K_cond=K_cond, BCid=BCid_t)
        elif (objectives[obj_flag] == "coupled_heat"):
            model = HeatCouplingGroup(
                fea_solver=fea_solver,
                lsm_solver=lsm_solver,
                nelx=nelx,
                nely=nely,
                force_e=GF_e,
                force_t=GF_t,
                movelimit=movelimit,
                K_cond=K_cond,
                BCid_e=BCid_e,
                BCid_t=BCid_t,
                E=E, nu=nu, alpha=alpha,
                w=0.0) # if w = 0.0, thermoelastic + conduction, if w = 1.0, conduction only

        # One Problem per one OpenMDAO object
        prob = Problem(model)

        # optimize ...
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'IPOPT'
        prob.driver.opt_settings['linear_solver'] = 'ma27'

        prob.setup(check=False)
        prob.run_model()

        # Total derivative using MAUD =====================
        total = prob.compute_totals()
        if (objectives[obj_flag] == "compliance"):
            Cf = total['compliance_comp.compliance', 'inputs_comp.Vn'][0]
            Cg = total['weight_comp.weight', 'inputs_comp.Vn'][0]
        elif (objectives[obj_flag] == "stress"):
            Cf = total['pnorm_comp.pnorm', 'inputs_comp.Vn'][0]
            Cg = total['weight_comp.weight', 'inputs_comp.Vn'][0]
        elif (objectives[obj_flag] == "conduction"):
            Cf = total['compliance_comp.compliance', 'inputs_comp.Vn'][0]
            Cg = total['weight_comp.weight', 'inputs_comp.Vn'][0]
        elif (objectives[obj_flag] == "coupled_heat"):
            Cf = total['objective_comp.y', 'inputs_comp.Vn'][0]
            Cg = total['weight_comp.weight', 'inputs_comp.Vn'][0]

        nBpts = int(bpts_xy.shape[0])
        Cf = -Cf[:nBpts]
        Cg = -Cg[:nBpts]

        Sf = np.divide(Cf, seglength)
        Sg = np.divide(Cg, seglength)

        # confine Sg
        Sg[Sg < - 1.5] = -1.5
        Cg = np.multiply(Sg, seglength)

        ########################################################
        ############## 		suboptimize 		################
        ########################################################
        if 0:
            suboptim = Solvers(bpts_xy=bpts_xy, Sf=Sf, Sg=Sg, Cf=Cf, Cg=Cg, length_x=length_x,
                            length_y=length_y, areafraction=areafraction, movelimit=movelimit)
            # suboptimization
            if 1:  # simplex
                Bpt_Vel = suboptim.simplex(isprint=False)
            else:  # bisection..
                Bpt_Vel = suboptim.bisection(isprint=False)
            timestep = 1.0

        elif 1: # works okay now.
            bpts_sens = np.zeros((nBpts,2))
            # issue: scaling problem
            #
            bpts_sens[:,0] = Sf
            bpts_sens[:,1] = Sg

            lsm_solver.set_BptsSens(bpts_sens)
            scales = lsm_solver.get_scale_factors()
            (lb2,ub2) = lsm_solver.get_Lambda_Limits()
            constraint_distance = (0.4 * nelx * nely) - areafraction.sum()

            model = LSM2D_slpGroup(lsm_solver = lsm_solver, num_bpts = nBpts, ub = ub2, lb = lb2,
                Sf = bpts_sens[:,0], Sg = bpts_sens[:,1], constraintDistance = constraint_distance, movelimit=movelimit)

            subprob = Problem(model)
            subprob.setup()

            subprob.driver = ScipyOptimizeDriver()
            subprob.driver.options['optimizer'] = 'SLSQP'
            subprob.driver.options['disp'] = True
            subprob.driver.options['tol'] = 1e-10

            subprob.run_driver()
            lambdas = subprob['inputs_comp.lambdas']
            displacements_ = subprob['displacement_comp.displacements']

            displacements_[displacements_ > movelimit] = movelimit
            displacements_[displacements_ < -movelimit] = -movelimit
            timestep =  1.0 #abs(lambdas[0]*scales[0])

            Bpt_Vel = displacements_ / timestep
            # print(timestep)
            del subprob

        else: # branch: perturb-suboptim
            bpts_sens = np.zeros((nBpts,2))
            # issue: scaling problem
            #
            bpts_sens[:,0] = Sf
            bpts_sens[:,1] = Sg

            lsm_solver.set_BptsSens(bpts_sens)
            scales = lsm_solver.get_scale_factors()
            (lb2,ub2) = lsm_solver.get_Lambda_Limits()

            constraint_distance = (0.4 * nelx * nely) - areafraction.sum()
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
            displacements_[displacements_ > movelimit] = movelimit
            displacements_[displacements_ < -movelimit] = -movelimit
            timestep =  1.0 #abs(lambdas[0]*scales[0])
            Bpt_Vel = displacements_ / timestep
            # scaling
            # Bpt_Vel = Bpt_Vel#/np.max(np.abs(Bpt_Vel))

        lsm_solver.advect(Bpt_Vel, timestep)
        lsm_solver.reinitialise()

        if not inspctFlag: # quickplot
            plt.figure(1)
            plt.clf()
            plt.scatter(bpts_xy[:, 0], bpts_xy[:, 1], 10)
            plt.axis("equal")
            plt.savefig(loadFolder + 'restart_' + str(restart_iter) + "/" + "figs/bpts_%d.png" % i_HJ)

        print ('loop %d is finished' % i_HJ)
        area = areafraction.sum()/(nelx*nely)
        try:
            u = prob['temp_comp.disp']
            compliance = np.dot(u, GF_t[:nNODE])
        except:
            u = prob['disp_comp.disp']
            # compliance = np.dot(u, GF_e[:nDOF_e])
            pass

            if 1:  # quickplot
                plt.figure(1)
                plt.clf()
                plt.scatter(bpts_xy[:, 0], bpts_xy[:, 1], 10)
                plt.axis("equal")
                plt.savefig(loadFolder + 'restart_' + str(restart_iter) + "/" + "figs/bpts_%d.png" % i_HJ)
                if obj_flag == 3 or obj_flag == 2:
                    plt.figure(2)
                    plt.clf()
                    [xx, yy] = np.meshgrid(range(0,161),range(0,81))
                    plt.contourf(xx, yy,np.reshape(u, [81,161]))
                    plt.colorbar()
                    plt.axis("equal")
                    plt.scatter(bpts_xy[:, 0], bpts_xy[:, 1], 5)
                    plt.savefig(loadFolder + 'restart_' + str(restart_iter) + "/" + "figs/temp_%d.png" % i_HJ)

        if (objectives[obj_flag] == "compliance" and not inspctFlag):
            
            compliance = prob['compliance_comp.compliance']
            print (compliance, area)

            fid = open(loadFolder + 'restart_' + str(restart_iter) + "/" + "log.txt", "a+")
            fid.write(str(compliance) + ", " + str(area) + "\n")
            fid.close()
        elif (objectives[obj_flag] == "stress" and not inspctFlag):
            print (prob['pnorm_comp.pnorm'][0], area)

            fid = open(loadFolder + 'restart_' + str(restart_iter) + "/" + "log.txt", "a+")
            fid.write(str(prob['pnorm_comp.pnorm'][0]) +
                      ", " + str(area) + "\n")
            fid.close()
        elif (objectives[obj_flag] == "coupled_heat" and not inspctFlag):
            obj1 = prob['objective_comp.x1'][0]
            obj2 = prob['objective_comp.x2'][0]
            obj = prob['objective_comp.y'][0]

            print([obj1, obj2, obj,  area])
            fid = open(loadFolder + 'restart_' + str(restart_iter) + "/" + "log.txt", "a+")
            fid.write(str(obj1) + ", " + str(obj2) + ", " +
                      str(obj) + ", " + str(area) + "\n")
            fid.close()

        # Saving Phi
        phi = lsm_solver.get_phi()

        if not inspctFlag:
            raw = {}
            raw['phi'] = phi
            filename = loadFolder + 'restart_' + str(restart_iter) + '/' + 'phi%03i.pkl' % i_HJ
            with open(filename, 'wb') as f:
                pickle.dump(raw, f)

        del model
        del prob

        mem = virtual_memory()
        print (str(mem.available/1024./1024./1024.) + "GB")
        if mem.available/1024./1024./1024. < 3.0:
            print("memory explodes at iteration %3i " % i_HJ)
            exit()

# if __name__ == '__main__':
#     main(500)
# else:
#     main(-1)  # inspection mode
