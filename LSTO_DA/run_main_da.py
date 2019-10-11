# A cleaned version of run.
# no subtantial change, but make the routine more readable

from openmdao.api import Group, Problem, view_model, pyOptSparseDriver, ScipyOptimizeDriver
from openmdao.api import IndepVarComp, ExplicitComponent, ImplicitComponent
from post.plot import get_mesh, plot_solution, plot_contour

import cPickle as pickle
import numpy as np
from psutil import virtual_memory

# imports Cython wrappers for OpenLSTO_FEA, OpenLSTO_LSM
from pyBind import py_FEA
from py_lsmBind import py_LSM

# imports perturbation method (aka discrete adjoint)
from groups.PerturbGroup import *
from groups.lsm2d_SLP_Group_openlsto import LSM2D_slpGroup

# imports solvers for suboptimization
# TODO: needs to be replaced with OpenMDAO optimizer
from suboptim.solvers import Solvers
import scipy.optimize as sp_optim

objectives = {0: "compliance", 1: "stress",
              2: "conduction", 3: "coupled_heat"}
saveFolder = "./save/"
import os
try:
    os.mkdir(saveFolder)
except:
    pass
try:
    os.mkdir(saveFolder + 'figs')
except:
    pass

def main(maxiter):

    # select which problem to solve
    obj_flag = 3
    print(locals())
    print("solving %s problem" % objectives[obj_flag])

    ########################################################
    ################# 		FEA 		####################
    ########################################################
    # NB: only Q4 elements + integer-spaced mesh are assumed
    nelx = 160
    nely = 80

    length_x = 160.
    length_y = 80.

    ls2fe_x = length_x/float(nelx)
    ls2fe_y = length_y/float(nely)

    num_nodes_x = nelx + 1
    num_nodes_y = nely + 1

    nELEM = nelx * nely
    nNODE = num_nodes_x * num_nodes_y

    # NB: nodes for plotting (quickfix...)
    nodes = get_mesh(num_nodes_x, num_nodes_y, nelx, nely)

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
    E = 1.
    nu = 0.3
    f = -1  # dead load
    K_cond = 0.1  # thermal conductivity
    alpha = 1e-5  # thermal expansion coefficient

    fea_solver.set_material(E=E, nu=nu, rho=1.0)

    # Boundary Conditions =====================================
    if 1:
        coord_e = np.array([[0., 0.], [length_x, 0.]])
        tol_e = np.array([[1e-3, 1e3], [1e-3, 1e+3]])
        fea_solver.set_boundary(coord=coord_e, tol=tol_e)

        BCid_e = fea_solver.get_boundary()
        nDOF_e_wLag = nDOF_e + len(BCid_e)  # elasticity DOF
        
        coord = np.array([length_x*0.5, 0.0])  # length_y])
        tol = np.array([4.1, 1e-3])
        GF_e_ = fea_solver.set_force(coord=coord, tol=tol, direction=1, f=-f)
        GF_e = np.zeros(nDOF_e_wLag)
        GF_e[:nDOF_e] = GF_e_
    else: # cantilever bending
        coord_e = np.array([[0,0]])
        tol_e = np.array([[1e-3,1e10]])
        fea_solver.set_boundary(coord = coord_e,tol = tol_e)
        BCid_e = fea_solver.get_boundary()
        nDOF_e_wLag = nDOF_e + len(BCid_e)  # elasticity DOF

        coord = np.array([length_x,length_y/2])
        tol = np.array([1,1]) 
        GF_e_ = fea_solver.set_force(coord = coord,tol = tol, direction = 1, f = -1.0)
        GF_e = np.zeros(nDOF_e_wLag)
        GF_e[:nDOF_e] = GF_e_

    xlo = np.array(range(0, nNODE, num_nodes_x))
    xhi = np.array(range(nelx, nNODE, num_nodes_x))
    # xfix = np.array([num_nodes_x*(nely/2-1), num_nodes_x*nely/2,
                    # num_nodes_x*(nely/2-1) + nelx, num_nodes_x*nely/2 + nelx])
    xfix = np.append(xlo, xhi)
    yfix = np.array(range(num_nodes_x*nely, nNODE))
    # yloid = np.array(range(70, 91))
    fixID_d = np.append(xfix, yfix)
    fixID_d = np.unique(fixID_d)
    fixID = np.append(fixID_d, arange(70, 91))
    BCid_t = np.array(fixID, dtype=int)
    nDOF_t_wLag = nDOF_t + len(BCid_t)  # temperature DOF (zero temp)

    GF_t = np.zeros(nDOF_t_wLag)  # FORCE_HEAT (NB: Q matrix)
    # for ee in range(70, 91):  # between element 70 to 91
    #     GF_t[elem[ee]] += 10.  # heat generation
    # GF_t /= np.sum(GF_t)
    GF_t[nDOF_t:nDOF_t+len(fixID_d)+1] = 100.
        # GF_t[:] = 0.0


    ########################################################
    ################# 		LSM 		####################
    ########################################################
    movelimit = 0.5

    # Declare Level-set object
    lsm_solver = py_LSM(nelx=nelx, nely=nely, moveLimit=movelimit)

    if ((nelx == 160) and (nely == 80)):  # 160 x 80 case
        hole = array([[16, 14, 5], [48, 14, 5], [80, 14, 5],
                      [112, 14, 5], [144, 14, 5], [32, 27, 5],
                      [64, 27, 5], [96, 27, 5], [128, 27, 5],
                      [16, 40, 5], [48, 40, 5], [80, 40, 5],
                      [112, 40, 5], [144, 40, 5], [32, 53, 5],
                      [64, 53, 5], [96, 53, 5], [128, 53, 5],
                      [16, 66, 5], [48, 66, 5], [80, 66, 5],
                      [112, 66, 5], [144, 66, 5]], dtype=float)

        # NB: level set value at the corners should not be 0.0
        hole = append(hole, [[0., 0., 0.1], [0., 80., 0.1], [
            160., 0., 0.1], [160., 80., 0.1]], axis=0)

        lsm_solver.add_holes(locx=list(hole[:, 0]), locy=list(
            hole[:, 1]), radius=list(hole[:, 2]))

    elif ((nelx == 80) and (nely == 40)):
        hole = np.array([[8, 7, 2.5], [24, 7, 2.5], [40, 7, 2.5],
                         [56, 7, 2.5], [72, 7, 2.5], [16, 13.5, 2.5],
                         [32, 13.5, 2.5], [48, 13.5, 2.5], [64, 13.5, 2.5],
                         [8, 20, 2.5], [24, 20, 2.5], [40, 20, 2.5],
                         [56, 20, 2.5], [72, 20, 2.5], [16, 26.5, 2.5],
                         [32, 26.5, 2.5], [48, 26.5, 2.5], [64, 26.5, 2.5],
                         [8, 33, 2.5], [24, 33, 2.5], [40, 33, 2.5],
                         [56, 33, 2.5], [72, 33, 2.5]], dtype=np.float)

        # NB: level set value at the corners should not be 0.0
        hole = append(hole, [[0., 0., 0.1], [0., 40., 0.1], [
            80., 0., 0.1], [80., 40., 0.1]], axis=0)

        lsm_solver.add_holes(locx=list(hole[:, 0]), locy=list(
            hole[:, 1]), radius=list(hole[:, 2]))

    else:
        lsm_solver.add_holes([], [], [])

    lsm_solver.set_levelset()

    for i_HJ in range(maxiter):
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

        # bracketing Sf and Sg
        Sg[Sg < - 1.5] = -1.5
        Cg = np.multiply(Sg, seglength)

        # Sf[Sf*0.5 > movelimit] = movelimit/0.5
        # Sf[Sf*0.5 < -movelimit] = -movelimit/0.5
        # Cf = np.multiply(Sf, seglength)
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

            # save - wip =====================================
            plt.figure(1)
            plt.clf()
            plt.subplot(2,1,1)
            plt.scatter(bpts_xy[:,0], bpts_xy[:,1], 10, Bpt_Vel) 
            plt.subplot(2,1,2)
            plt.plot(Bpt_Vel)
            plt.savefig("/home/hayoung/Desktop/vel.png")
            # exit()
            # save - wip =====================================

        elif 1: # branch: perturb-suboptim
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

            # save - wip =====================================
            plt.figure(1)
            plt.clf()
            plt.subplot(2,1,1)
            plt.scatter(bpts_xy[:,0], bpts_xy[:,1], 10, Bpt_Vel) 
            plt.subplot(2,1,2)
            plt.plot(Bpt_Vel)
            plt.savefig("/home/hayoung/Desktop/vel.png")
            # exit()
            # save - wip =====================================

        else: # branch: perturb-suboptim
            bpts_sens = np.zeros((nBpts,2))
            # issue: scaling problem
            # 
            bpts_sens[:,0] = Sf
            bpts_sens[:,1] = Sg

            # save - wip =====================================
            plt.figure(1)
            plt.clf()
            plt.subplot(2,1,1)
            plt.scatter(bpts_xy[:,0], bpts_xy[:,1], 10, bpts_sens[:,0])
            plt.subplot(2,1,2)
            plt.plot(bpts_sens[:,0])
            plt.savefig("/home/hayoung/Desktop/main_sens.png")
            # save - wip =====================================

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

            # save - wip =====================================
            np.savetxt("/home/hayoung/Desktop/disp.txt", bpts_sens)
            plt.figure(1)
            plt.clf()
            plt.subplot(2,1,1)
            plt.scatter(bpts_xy[:,0], bpts_xy[:,1], 10, displacements_) 
            plt.subplot(2,1,2)
            plt.plot(displacements_)
            plt.savefig("/home/hayoung/Desktop/disp.png")
            # exit()
            # save - wip =====================================
            Bpt_Vel = displacements_ / timestep
            # scaling 
            # Bpt_Vel = Bpt_Vel#/np.max(np.abs(Bpt_Vel))

        lsm_solver.advect(Bpt_Vel, timestep)
        lsm_solver.reinitialise()

        print ('loop %d is finished' % i_HJ)
        area = areafraction.sum()/(nelx*nely)
        try: 
            u = prob['temp_comp.disp']
            compliance = np.dot(u, GF_t[:nNODE])
        except:
            u = prob['disp_comp.disp']
            compliance = np.dot(u, GF_e[:nDOF_e])

        if 1:  # quickplot
            plt.figure(1)
            plt.clf()
            plt.scatter(bpts_xy[:, 0], bpts_xy[:, 1], 10)
            plt.axis("equal")
            plt.savefig(saveFolder + "figs/bpts_%d.png" % i_HJ)
            if obj_flag == 3:
                plt.figure(2)
                plt.clf()
                [xx, yy] = np.meshgrid(range(0,161),range(0,81))
                plt.contourf(xx, yy,np.reshape(u, [81,161]))
                plt.colorbar()
                plt.axis("equal")
                plt.scatter(bpts_xy[:, 0], bpts_xy[:, 1], 5)
                plt.savefig(saveFolder + "figs/temp_%d.png" % i_HJ)

        # print([compliance[0], area])
        if (objectives[obj_flag] == "compliance"):
            compliance = prob['compliance_comp.compliance']
            print (compliance, area)

            fid = open(saveFolder + "log.txt", "a+")
            fid.write(str(compliance) + ", " + str(area) + "\n")
            fid.close()
        elif (objectives[obj_flag] == "stress"):
            print (prob['pnorm_comp.pnorm'][0], area)

            fid = open(saveFolder + "log.txt", "a+")
            fid.write(str(prob['pnorm_comp.pnorm'][0]) +
                      ", " + str(area) + "\n")
            fid.close()
        elif (objectives[obj_flag] == "coupled_heat"):
            obj1 = prob['objective_comp.x1'][0]
            obj2 = prob['objective_comp.x2'][0]
            obj = prob['objective_comp.y'][0]
            
            print([obj1, obj2, obj,  area])
            fid = open(saveFolder + "log.txt", "a+")
            fid.write(str(obj1) + ", " + str(obj2) + ", " +
                      str(obj) + ", " + str(area) + "\n")
            fid.close()

        # Saving Phi
        phi = lsm_solver.get_phi()

        if i_HJ == 0:
            raw = {}
            raw['mesh'] = nodes
            raw['nodes'] = nodes
            raw['elem'] = elem
            raw['GF_e'] = GF_e
            raw['GF_t'] = GF_t
            raw['BCid_e'] = BCid_e
            raw['BCid_t'] = BCid_t
            raw['E'] = E
            raw['nu'] = nu
            raw['f'] = f
            raw['K_cond'] = K_cond
            raw['alpha'] = alpha
            raw['nelx'] = nelx
            raw['nely'] = nely
            raw['length_x'] = length_x
            raw['length_y'] = length_y
            raw['coord_e'] = coord_e
            raw['tol_e'] = tol_e
            filename = saveFolder + 'const.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(raw, f)

        raw = {}
        raw['phi'] = phi
        filename = saveFolder + 'phi%03i.pkl' % i_HJ
        with open(filename, 'wb') as f:
            pickle.dump(raw, f)

        del model
        del prob

        mem = virtual_memory()
        print (str(mem.available/1024./1024./1024.) + "GB")
        if mem.available/1024./1024./1024. < 3.0:
            print("memory explodes at iteration %3i " % i_HJ)
            return()

if __name__ == "__main__":
    main(300)
else:
    main(1)  # testrun
