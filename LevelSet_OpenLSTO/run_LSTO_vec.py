import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sys

from openmdao.api import Problem, view_model, ScipyOptimizer

from fem2d import PyFEMSolver
from forces import get_forces

from lsm2d_SLP_Group import LSM2D_slpGroup
from lsm_classes import PyLSMSolver

from ls_sensitivity import _LeastSquare

import matplotlib.pyplot as plt
import scipy.optimize as sp_optim

from optim_refact_v3 import Cantilever


# FEM Mesh
num_nodes_x = 81  # 161
num_nodes_y = 41  # 81
num_nodes = num_nodes_x * num_nodes_y
num_elems = (num_nodes_x - 1) * (num_nodes_y - 1)

# LSM Mesh
num_param_x = num_nodes_x
num_param_y = num_nodes_y

length_x = num_nodes_x - 1
length_y = num_nodes_y - 1

num_dofs = num_nodes_x * num_nodes_y * 2 + num_nodes_y * 2

# FEA properties
E = 1.
nu = 0.3
f = -1.
fem_solver = PyFEMSolver(num_nodes_x, num_nodes_y,
                         length_x, length_y, E, nu)  # , True)
forces = get_forces(num_nodes_x, num_nodes_y, f=f)
rhs = np.zeros(num_dofs)
rhs[:num_nodes_x * num_nodes_y * 2] = forces

Order_gpts = 2
num_gpts = num_elems * Order_gpts**2


#num_gpts = 2

# LSM properties
radius = 2
movelimit = 0.5

# LSM initialize (swisscheese config)

lsm_solver = PyLSMSolver(num_nodes_x, num_nodes_y, 0.5)

# HJ loop
max_loop = 100
for i_HJ in range(0, max_loop):
    # 0. discretize
    (bpts_xy, areafraction, segmentLength) = lsm_solver.discretize()
    if i_HJ == 0:
        bpts_xy0 = bpts_xy
        areafraction0 = areafraction

    plt.clf()
    plt.plot(bpts_xy[:, 0], bpts_xy[:, 1], 'o')
    plt.savefig('plots3/bpts_%d.png' % i_HJ)

    num_sparse = num_elems * 64 * 4 + 2 * 2 * num_nodes_y
    irs = np.zeros(num_sparse, dtype=np.int32)
    jcs = np.zeros(num_sparse, dtype=np.int32)
    data = np.zeros(num_sparse)

    fem_solver.get_stiffness_matrix_LSTO(areafraction, data, irs, jcs)

    # 1. get stiffness matrix & solve (LU decomposition)
    mtx = scipy.sparse.csc_matrix(
        (data, (irs, jcs)), shape=(num_dofs, num_dofs))
    lumtx = scipy.sparse.linalg.splu(mtx)
    _dofs = lumtx.solve(rhs)
    u_fem2d = _dofs[:num_nodes * 2]

    # TESTING ++++++++++++++++++++++++++++
    # plot_solution(orig_nodes, deflected_nodes=u_fem2d)
    # QUICKFIX ===============================
    cantilever = Cantilever(True)
    cantilever.set_fea()

    Order_gpts = 2
    num_gpts = num_elems * Order_gpts**2

    u_cant = cantilever.get_u(areafraction)
    # ==========================================
    (IntegrationPointSensitivities,
     BoundarySensitivities) = cantilever.get_sens(bpts_xy)
    # +++++++++++++++++++++++++++++++++++++
    np.savetxt('./sens/cant_bpts_%d.txt' % i_HJ, BoundarySensitivities)
    np.savetxt('./sens/cant_Gpts_%d.txt' % i_HJ, IntegrationPointSensitivities)

    xpos = np.zeros(num_elems * 4)
    ypos = np.zeros(num_elems * 4)
    sens_compl = np.zeros(num_elems * 4)

    fem_solver.get_sensitivity_LSTO(
        u_fem2d, xpos, ypos, sens_compl)  # verified

    xpos = xpos.reshape((num_elems, Order_gpts**2))  # [:,(2,3,0,1)]
    ypos = ypos.reshape((num_elems, Order_gpts**2))  # [:,(2,3,0,1)]
    fixedGpts_sens = sens_compl.reshape(
        (num_elems, Order_gpts**2))  # [:,(2,3,0,1)]

    Gdot = np.zeros((num_elems, Order_gpts**2))
    xdot = np.zeros((num_elems, Order_gpts**2))
    ydot = np.zeros((num_elems, Order_gpts**2))

    for qq in range(0, 4):
        tmp_v = fixedGpts_sens[:, qq].reshape((40, 80))
        Gdot[:, qq] = tmp_v.transpose().flatten()
        tmp_v = xpos[:, qq].reshape((40, 80))
        xdot[:, qq] = tmp_v.transpose().flatten()
        tmp_v = ypos[:, qq].reshape((40, 80))
        ydot[:, qq] = tmp_v.transpose().flatten()

    fixedGpts_xy = np.zeros((num_elems, Order_gpts**2, 2))
    fixedGpts_xy[:, :, 0] = xdot  # xpos
    fixedGpts_xy[:, :, 1] = ydot  # ypos

    leastsquare = _LeastSquare(bpts_xy, fixedGpts_xy,
                               Gdot, areafraction, radius)  # fixedGpts_sens, areafraction, radius)
    # leastsquare = _LeastSquare(bpts_xy, cantilever.get_IntegPoints(),
    #             IntegrationPointSensitivities, areafraction, radius) # testing
    bpts_sens = leastsquare.get_sens_compliance()
    # sys.exit(0)
    na = np.zeros((40 * 80 * 4, 2))
    for ee in range(0, 40 * 80):
        for gg in range(0, 4):
            na[ee * 4 + gg, 0] = fixedGpts_xy[ee, gg, 0]
            na[ee * 4 + gg, 1] = fixedGpts_xy[ee, gg, 1]

    xy = cantilever.get_IntegPoints_xy()

    np.savetxt('./sens3/cant_xy_%d.txt' % i_HJ, xy)
    np.savetxt('./sens3/lsq_xy_%d.txt' % i_HJ, na)

    np.savetxt('./sens3/lsq_bpts_%d.txt' % i_HJ, bpts_sens)
    np.savetxt('./sens3/lsq_Gpts_%d.txt' % i_HJ, fixedGpts_sens)

    lambdas = np.zeros(2)
    lambdas = lsm_solver.preprocess(lambdas, movelimit, bpts_sens)
    (ub, lb) = lsm_solver.get_bounds()
    # (a,b,c,d) = lsm_solver.get_optimPars()
    # print (a,b,c,d)
    # ## start of the slp suboptimization

    if 0:
        def objF_nocallback(x):
            displacement = lsm_solver.computeDisplacements(x)
            # print('objF')
            # print(displacement)
            displacement_np = np.asarray(displacement)
            return lsm_solver.computeFunction(displacement_np, 0)[0]

        def conF_nocallback(x):
            displacement = lsm_solver.computeDisplacements(x)
            # print('conF')
            # print(displacement)
            displacement_np = np.asarray(displacement)
            return lsm_solver.computeFunction(displacement_np, 1)[1]

        cons = ({'type': 'eq', 'fun': lambda x: conF_nocallback(x)})
        res = sp_optim.minimize(objF_nocallback, np.zeros(2), method='SLSQP', options={'disp': True},
                                bounds=((lb[0], ub[0]), (lb[1], ub[1])),
                                constraints=cons)

        lambdas = res.x
        print(lambdas)

    if 1:
        model = LSM2D_slpGroup(
            lsm_solver=lsm_solver,
            bpts_xy=bpts_xy,  # boundary points
            lb=lb, ub=ub,
        )
        # model.approx_total_derivs(method = 'fd')

        prob = Problem(model)
        prob.setup()

        # if (i_HJ==0):
        #    view_model(prob)

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['disp'] = True

        prob.run_driver()

        lambdas = prob['inputs_comp.lambdas']
        print (lambdas)

        # totals = prob.compute_total_derivs(['objective_comp.objective','displacement_comp.displacements'],['inputs_comp.lambdas','displacement_comp.displacements'])
        # print (totals['objective_comp.objective','inputs_comp.lambdas'])
        # print (totals['objective_comp.objective','displacement_comp.displacements'])
        # print (totals['displacement_comp.displacements','inputs_comp.lambdas'])

    # after sub-optimization
    # print ('loop %d is finished & postprocessing starts' % i_HJ)
    lambdas = lsm_solver.postprocess(lambdas)
    # print(lambdas)
    # print ('computing velocities')
    lsm_solver.computeVelocities()
    print(lambdas)
    print ('starts updating')
    phi = lsm_solver.update(np.abs(lambdas[0]))
    plt.clf()
    plt.plot(phi, 'o')
    plt.savefig('plots3/phi_%d.png' % i_HJ)

    # print ('reinitialiing')
    lsm_solver.reinitialise()
    # (a,b,c,d) = lsm_solver.get_optimPars()
    # print (a,b,c,d)
    # lsm_solver.del_optim()

    # print ('all finished')
    print ('loop %d is finished' % i_HJ)

    # totals = prob.compute_total_derivs(['objective_comp.objective'], ['inputs_comp.lambdas'])
