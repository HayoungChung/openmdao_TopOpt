import sys
sys.path.append("/home/hac/Dropbox/packages/08.OpenMDAO_OpenLSTO/LSTO_DiscreteAdjoint/LSTO_AD/")
print(sys.path)
from pyBind import py_FEA
from py_lsmBind import py_LSM
import numpy as np

import time
import psutil

import gc

def mem():
    print(str(round(psutil.Process().memory_info().rss/1024./1024., 2)) + 'MB')

# @profile
def main():
    mem()
    print("fea")
    fea_solver = py_FEA(lx = 160, ly= 80, nelx = 160, nely = 80, element_order=2)
    mem()
    print("getmesh")
    a = fea_solver.get_mesh()
    mem()
    print('del')
    del fea_solver
    mem()
    del a 
    mem()

    print("fea")
    fea_solver2 = py_FEA(lx = 160, ly= 80, nelx = 160, nely = 80, element_order=2)
    mem()
    print("getmesh")
    b = fea_solver2.get_mesh()
    mem()
    print('del')
    del fea_solver2
    mem()
    del b 
    mem()

    print("fea")
    fea_solver3 = py_FEA(lx = 160, ly= 80, nelx = 160, nely = 80, element_order=2)
    mem()
    print("getmesh")
    fea_solver3.get_mesh()
    mem()
    print('del')
    del fea_solver3
    mem()


    print("lsm")
    lsm_solver = py_LSM(nelx = 160, nely = 80, moveLimit = 0.5)
    mem()
    print("set_levelset")
    lsm_solver.set_levelset()
    mem()
    # lsm_solver.set_levelset()
    # mem()
    # lsm_solver.set_levelset()
    # mem()
    print("discretise")
    lsm_solver.discretise()
    mem()
    print("freeing boundaryptr")
    lsm_solver.freeing_pointers()
    mem()
    print("discretise")
    lsm_solver.discretise()
    mem()
    # lsm_solver.discretise()
    # mem()
    print("freeing boundaryptr")
    lsm_solver.freeing_pointers()
    mem()
    print("discretise")
    (bpts_xy, areafraction, seglength) = lsm_solver.discretise()
    mem()
    print("computeVn")
    Bpt_Vel = np.zeros(bpts_xy.shape[0])
    mem()
    print("Advect")
    lsm_solver.advect(Bpt_Vel, 1.0)
    mem()
    print("freeing boundaryptr")
    lsm_solver.freeing_pointers()
    mem()
    print("reinit")
    lsm_solver.reinitialise()
    mem()
    
    print("dealloc")
    lsm_solver.dealloc()
    mem()
    
    print("deleting")
    del lsm_solver
    mem()
    # del fea_solver
    # mem()
    time.sleep(1.0)
    mem()



if __name__ == "__main__":
    main()