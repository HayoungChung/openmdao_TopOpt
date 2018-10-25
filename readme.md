# Installation 

## Prerequisite

OpenMDAO ver. 2.2
> Originally the program is developed based on OpenMDAO ver. 2.0, so you'll see some *DeprecationWarning* messages.

## Make shared library for FE analysis from OpenLSTO

Following operations are assumed to take place in the default path. (*M2DO_PATH*)

1. Move to M2DO_FEA/Python, where Cython wrapper (pyBind.pyx) is located at

```
(at M2DO_PATH)
cd ./OpenLSTO-master/M2DO_FEA/Python/
```

2. Compile and copy *.so file to OpenMDAO-SIMP path

```
sh makefile.sh 
```
or 
```
python FEAsetup.py build
cp build/lib*/*.so ./../../../Density*
```

3. Run main python script

```
(at M2DO_PATH)
cd Density_OpenLSTO/
python run_openmdao.py
```

4. (Optional) Change mesh density

In run_openmdao.py, change 

```
nelx = 40 # number of elements in x-direction
nely = 20 # number of elements in y-direction
```

