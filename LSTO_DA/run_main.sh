## NOT WORKING!! ##
#!/bin/bash

python run_main_da.py &&
workingF="./save_coupleHeat1/" # folder WIP

END=5
for T in {1..$END}
do
    q=`ls phi*.pkl|wc -l`;echo $q
    workingF="${workginF}/restart_$"