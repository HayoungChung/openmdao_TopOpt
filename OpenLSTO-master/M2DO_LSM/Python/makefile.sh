#!/bin/bash 

python LSMsetup.py build
cp build/lib*/*.so ./
cp build/lib*/*.so ./../../../LevelSet*/ 
