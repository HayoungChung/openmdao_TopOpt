# Introduction
This program presents a modular paradigm for topology optimization using OpenMDAO, an open-source computational framework for multidisciplinary design optimization.

# Installation 

## Prerequisite

* gcc version >= 4.7 
* OpenMDAO ver. >= 2.2
> Originally the program is developed based on OpenMDAO ver. 2.0, so you'll see some *DeprecationWarning* messages.

## Compiling FEA shared library (*.so) from OpenLSTO for FE analysis 

Following operations are assumed to take place in the default path. (*M2DO_PATH*)

1. Move to M2DO_FEA/Python, where Cython wrapper (pyBind.pyx) is located at

    ```
    (at M2DO_PATH)
    cd ./OpenLSTO-master/M2DO_FEA/Python/
    ```

3. Compile and copy *.so file to OpenMDAO-SIMP path
    ```
    sh makefile.sh 
    ```
    or 
    ```
    python FEAsetup.py build
    cp build/lib*/*.so ./../../../Density*
    cp build/lib*/*.so ./../../../LevelSet*/ 
    ```

## Compiling LSM shared library (*.so) from OpenLSTO for level-set operations 

Following operations are assumed to take place in the default path. (*M2DO_PATH*)

1. Move to M2DO_LSM/Python, where Cython wrapper (py_lsmBind.pyx) is located at

    ```
    (at M2DO_PATH)
    cd ./OpenLSTO-master/M2DO_LSM/Python/
    ```

2. Compile and copy *.so file to OpenMDAO-SIMP path
    ```
    sh makefile.sh 
    ```
    or 
    ```
    python LSMsetup.py build
    cp build/lib*/*.so ./../../../LevelSet* 
    ```
    
# Running a SIMP optimization

1. Run main python script

    ```
    (at M2DO_PATH)
    cd Density_OpenLSTO/
    python run_openmdao.py
    ```

4. (Optional) Changing a mesh density

    In run_openmdao.py, change 

    ```
    nelx = 40 # number of elements in x-direction
    nely = 20 # number of elements in y-direction
    ```

5. Visualize the results

    ```
    python make_plots.py
    ```

# Running a LSTO optimization

1. Run main python script

    ```
    (at M2DO_PATH)
    cd LevelSet_OpenLSTO/
    python run_LSTO_openLSTO.py
    ```

4. (Optional) Changing a mesh density

    In run_LSTO_openLSTO.py, change 

    ```
    nelx = 40 # number of elements in x-direction
    nely = 20 # number of elements in y-direction
    ```

5. Visualize the results

    ```
    python make_plots.py
    ```
# References

1. OpenLSTO-lite

    Original FEA and LSM routines are derived from [OpenLSTO](https://github.com/M2DOLab/OpenLSTO-lite), which is am open-source level-set topology optimization program developed and maintained by researchers of [M2DO group](http://m2do.ucsd.edu/) led by Prof. Alicia Kim.

    Comprehensive documentation is found in the [webpage](http://m2do.ucsd.edu/software/)

2. OpenMDAO

    The Present program is based on [OpenMDAO](http://openmdao.org/), an open-source high-performance computing platform for systems analysis and multidisciplinary optimization, written in Python. 

3. SLSM library

    Implementation details of the level-set method used herein can be found in [SLSM library](https://github.com/lohedges/slsm) developed my Dr. Lester Hedges. 
