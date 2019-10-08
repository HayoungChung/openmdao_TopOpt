from libcpp.vector cimport vector
from cpython cimport array
from libcpp cimport bool, int

from py_mesh cimport Mesh
from py_hole cimport Hole
from py_commons cimport BoundaryPoint, Coord

import numpy as np 
cimport numpy as np 

cdef extern from "./../include/mersenne_twister.h":
    cdef cppclass MersenneTwister:
        MersenneTwister() except +


cdef extern from "./../include/level_set.h":
    cdef cppclass LevelSet:
        LevelSet(Mesh&, double , unsigned int, bool) except +
        LevelSet(Mesh&, vector[Hole], double , unsigned int, bool) except +
        
        bool update(double)
        bool update_no_WENO(double)
        void reinitialise()
        void computeVelocities(vector[BoundaryPoint]&)        
        double computeVelocities(vector[BoundaryPoint]&, 
                        double&, const double, MersenneTwister&);
        void computeGradients()
        void killNodes(vector[Coord]&)
        void fixNodes(vector[Coord]&)

        vector[double] signedDistance
        vector[double] velocity
        vector[double] gradient
        vector[double] target

        double moveLimit
