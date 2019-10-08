# this main script runs "Conduction + thermoelasticity coupled" problem
from openmdao.api import Group, Problem, view_model, pyOptSparseDriver #ScipyOptimizeDriver
from openmdao.api import IndepVarComp, ExplicitComponent, ImplicitComponent
from post.plot import get_mesh, plot_solution, plot_contour
try:
    import cPickle as pickle
except:
    import pickle

from pylab import *
import numpy as np

from pyBind import py_FEA, py_Sensitivity
from py_lsmBind import py_LSM

from groups.PerturbGroup import *

from suboptim.solvers import Solvers

def main(itr):

    # perturb?
    isPerturb = True
    pertb = 0.2

    #problem?
    objectives={"Compliance":0, "Stress":1, "Conduction": 2, "CoupledHeat" : 3}
    obj_flag = 3

    # FEM Mesh
    if 1:
        nelx = 160
        nely = 80
    else: # checking partials
        nelx = 20
        nely = 10

    length_x = 160.
    length_y = 80.

    ls2fe_x = length_x/nelx
    ls2fe_y = length_y/nely

    num_nodes_x = nelx + 1
    num_nodes_y = nely + 1

    num_elems = (num_nodes_x - 1) * (num_nodes_y - 1)

    nodes = get_mesh(num_nodes_x, num_nodes_y, nelx, nely) # for plotting

    # FEA properties
    E = 1.
    nu = 0.3
    f = -1.
    K_cond = 1.
    alpha = 1e-5

    fea_solver = py_FEA(lx = length_x, ly = length_y, nelx=nelx, nely=nely, element_order=2)
    [node, elem, elem_dof] = fea_solver.get_mesh()

    nELEM = elem.shape[0]
    nNODE = node.shape[0]
    nDOF_t = nNODE * 1
    nDOF_e = nNODE * 2

    fea_solver.set_material(E=E,nu=nu,rho=1.0)

    ## BCs ===================================
    '''
    coord = np.array([[0.,0.]])
    tol = np.array([[1e-3,1e+3]])
    fea_solver.set_boundary(coord = coord,tol = tol)
    np.savetxt("BC0.txt", fea_solver.get_boundary())

    coord = np.array([[length_x,0.]])
    tol = np.array([[1e-3,1e+3]])
    fea_solver.set_boundary(coord = coord,tol = tol)
    np.savetxt("BC1.txt", fea_solver.get_boundary())
    '''
    coord = np.array([[0., 0.],[length_x,0.]])
    tol = np.array([[1e-3, 1e3],[1e-3,1e+3]])
    fea_solver.set_boundary(coord = coord,tol = tol)

    BCid_e = fea_solver.get_boundary()
    nDOF_e_withLag  = nDOF_e + len(BCid_e)

    xlo = np.array(range(0,161*81,161))
    xhi = np.array(range(160, 161*81, 161))
    xfix = np.append(xlo,xhi)
    BCid_t = np.array(xfix,dtype=int)
    nDOF_t_withLag  = nDOF_t + len(BCid_t)

    coord = np.array([length_x*0.5,0.0]) #length_y])
    tol = np.array([4.1, 1e-3])
    GF_e_ = fea_solver.set_force(coord = coord,tol = tol, direction = 1, f = 1.0)
    GF_e = np.zeros(nDOF_e_withLag)
    GF_e[:nDOF_e] = GF_e_

    GF_t = np.zeros(nDOF_t_withLag) # GF_HEAT
    for ee in range(nELEM):
        GF_t[elem[ee]] += 1.
    GF_t /= np.sum(GF_t)

    # LSM properties
    radius = 2
    movelimit = 0.1
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

    elif ((nelx == 80) and (nely == 40)):
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
    else:
        lsm_solver.add_holes([],[],[])
        # lsm_solver.add_holes(locx = list(hole[:,0]), locy = list(hole[:,1]), radius = list(hole[:,2]))
    lsm_solver.set_levelset()
    maxiter = 150
    for i_HJ in range(maxiter):
        (bpts_xy, areafraction, seglength) = lsm_solver.discretise()

        if (obj_flag == objectives["Compliance"]):
            model = ComplianceGroup(
                fea_solver = fea_solver,
                lsm_solver = lsm_solver,
                nelx = nelx,
                nely = nely,
                force = GF_e, movelimit = movelimit)
        elif (obj_flag == objectives["Stress"]):
            model = StressGroup(
                fea_solver = fea_solver,
                lsm_solver = lsm_solver,
                nelx = nelx,
                nely = nely,
                force = GF_e, movelimit = movelimit,
                pval = 5.0, E = E, nu = nu)
        elif (obj_flag == objectives["Conduction"]):
            model = ConductionGroup(
                fea_solver = fea_solver,
                lsm_solver = lsm_solver,
                nelx = nelx,
                nely = nely,
                force = GF_t, movelimit = movelimit,
                K_cond = K_cond, BCid = BCid_t)
        elif (obj_flag == objectives["CoupledHeat"]):
            model = HeatCouplingGroup(
                fea_solver = fea_solver,
                lsm_solver = lsm_solver,
                nelx = nelx,
                nely = nely,
                force_e = GF_e,
                force_t = GF_t,
                movelimit = movelimit,
                K_cond = K_cond,
                BCid_e = BCid_e,
                BCid_t = BCid_t,
                E = E, nu = nu, alpha = alpha,
                w = 0.0)


        prob = Problem(model)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'IPOPT'
        prob.driver.opt_settings['linear_solver'] = 'ma27'
        # prob.driver = ScipyOptimizeDriver()
        # prob.driver.options['optimizer'] = 'SLSQP'
        prob.setup(check=False)
        prob.run_model()

        #### TEMP_PLOT ####
        # checking temperatures distribution
        # SHOULD BE REMOVED AFTER IT IS CHECKED
        figure(2)
        clf()
        temp = prob['temp_comp.disp']
        xlin = np.linspace(0,160,161)
        ylin = np.linspace(0,80,81)
        xx, yy = np.meshgrid(xlin,ylin)
        temp_re = np.reshape(temp[:nNODE],[81,161])
        scatter(xx,yy,10, -1.0*np.multiply(temp_re, temp_re))
        axis('equal')
        colorbar()
        grid(True)
        box
        savefig('UtU.png')
        clf()
        # exit()
        #### TEMP_PLOT ####


        # checking temperature (VERIFIED with run_therm, 7/8/2019)
        # np.savetxt("temperature.txt", prob['temp_comp.disp'])
        # view_model(prob)
        # checking partials (VERIFIED, 7/5/2019)
        # prob.check_partials(includes=['conductivity_comp','coupledload_comp','elasticity_comp','objective_comp','weight_comp'], compact_print=True, step=1e-6)
        # exit()

        total = prob.compute_totals() # evoke solve_linear() once.
        if (obj_flag == objectives["Compliance"]):
            Sf = total['compliance_comp.compliance','inputs_comp.Vn']
            Sg = total['weight_comp.weight','inputs_comp.Vn']
        elif (obj_flag == objectives["Stress"]):
            Sf = total['pnorm_comp.pnorm','inputs_comp.Vn']
            Sg = total['weight_comp.weight','inputs_comp.Vn']
        elif (obj_flag == objectives["Conduction"]):
            Sf = total['compliance_comp.compliance','inputs_comp.Vn']
            Sg = total['weight_comp.weight','inputs_comp.Vn']
        elif (obj_flag == objectives["CoupledHeat"]):  # FIXME: sensivity is fairly estimated
            # Sf = total['compliance_et_comp.compliance','inputs_comp.Vn']
            # Sf = total['compliance_t_comp.compliance','inputs_comp.Vn'] 
            Sf = total['objective_comp.y','inputs_comp.Vn']
            Sg = total['weight_comp.weight','inputs_comp.Vn']

        nBpts = int(bpts_xy.shape[0])
        Sf = -Sf[0][:nBpts]
        Sg = -Sg[0][:nBpts]

        Cf = Sf
        Cg = Sg
        '''
        plt.figure(1)
        plt.clf()
        plt.scatter(bpts_xy[:,0],bpts_xy[:,1], 10, np.divide(Sf, seglength))
        plt.colorbar()
        plt.axis("equal")
        plt.savefig('sensivity.png')
        exit()
        '''
        suboptim = Solvers(bpts_xy=bpts_xy, Sf=Sf, Sg=Sg, Cf=Cf, Cg=Cg, length_x=length_x, length_y=length_y, areafraction=areafraction, movelimit=movelimit)
        # suboptimization
        if 1: # simplex
            Bpt_Vel = suboptim.simplex(isprint=False)

        else:  # bisection..
            Bpt_Vel = suboptim.bisection(isprint=False)

        timestep = 1.0

        lsm_solver.advect(Bpt_Vel, timestep)
        lsm_solver.reinitialise()

        if 1: # quick plot
            plt.figure(1)
            plt.clf()
            plt.scatter(bpts_xy[:,0],bpts_xy[:,1], 10)
            plt.axis("equal")
            plt.savefig("save_AD/figs/mdo_bpts_%d.png" % i_HJ)


        print ('loop %d is finished' % i_HJ)
        area = areafraction.sum()/(nelx*nely)
        u = prob['temp_comp.disp']
        compliance = np.dot(u,GF_t[:nNODE])
        obj1 = prob['objective_comp.x1'][0]
        obj2 = prob['objective_comp.x2'][0]
        obj  = prob['objective_comp.y'][0]

        print([compliance, area])
        if (obj_flag == objectives["Compliance"]):
            print (compliance, area)

            fid = open("save/log.txt","a+")
            fid.write(str(compliance) + ", " + str(area) + "\n")
            fid.close()
        elif (obj_flag == objectives["Stress"]):
            print (prob['pnorm_comp.pnorm'][0], area)

            fid = open("save/log.txt","a+")
            fid.write(str(prob['pnorm_comp.pnorm'][0]) + ", " + str(area) + "\n")
            fid.close()
        elif (obj_flag == objectives["CoupledHeat"]):
            print([obj1, obj2, obj,  area])
            fid = open("save_AD/log.txt","a+")
            fid.write(str(obj1) + ", " + str(obj2) + ", " + str(obj) + ", " + str(area) + "\n")
            fid.close()

        phi = lsm_solver.get_phi()

        if i_HJ == 0:
            raw = {}
            raw['mesh'] = nodes
            filename = 'save_AD/const.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(raw, f)

        raw = {}
        raw['phi'] = phi
        filename = 'save_AD/phi%03i.pkl' % i_HJ
        with open(filename, 'wb') as f:
            pickle.dump(raw, f)

if __name__ == "__main__":
    main(0)
