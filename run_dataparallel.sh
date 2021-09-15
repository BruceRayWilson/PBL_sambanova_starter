#!/bin/sh
# Stop on error -not
set e

#######################
# Edit these variables.
#######################
NUM_THREADS="export OMP_NUM_THREADS=1"
MODEL_NAME="FFNLogReg"
#######################
# Start script timer
SECONDS=0
# Temp file location
#DIRECTORY=$$
#OUTDIR=${HOME}/${DIRECTORY}

alias snpath='export PATH=$PATH:/opt/sambaflow/bin'
alias snthreads=$NUM_THREADS
alias snvenv='source /opt/sambaflow/venv/bin/activate'
alias snp='snpath;snthreads;snvenv'

snp

export SF_RNT_TILE_AFFINITY=0xf0000000

#source /opt/sambaflow/venv/bin/activate
#cd ${HOME}
echo "Model: ${MODEL_NAME}"
echo "Date: " $(date +%m/%d/%y)
echo "Time: " $(date +%H:%M)

# Change this directory path as necessary.
cd ~/sambanova_starter

echo "COMPILE"
# ws = world_size and its value really doesn't matter for a compile.
COMMAND="python sn_boilerplate_main.py compile --data-parallel -ws 2 -b=1 --pef-name=sn_boilerplate --output-folder=pef"
echo "COMPILE COMMAND: $COMMAND"
eval $COMMAND

#python sn_boilerplate_main.py test --distributed-run --pef="pef/sn_boilerplate/sn_boilerplate.pef"

echo "RUN 0"
# DOES NOT WORK!!
#RUN 0 COMMAND: /opt/mpich-3.3.2/bin/mpirun -np 8 python sn_boilerplate_main.py run --data-parallel -ws 8 --reduce-on-rdu --pef='pef/sn_boilerplate/sn_boilerplate.pef'
#usage: sn_boilerplate_main.py [-h] {compile,run,test,measure-performance} ...
#sn_boilerplate_main.py: error: unrecognized arguments: -ws 8
COMMAND="/opt/mpich-3.3.2/bin/mpirun -np 8 python sn_boilerplate_main.py run --data-parallel -ws 8 --reduce-on-rdu --pef='pef/sn_boilerplate/sn_boilerplate.pef'"
echo "RUN 0 COMMAND: $COMMAND"
#eval $COMMAND

echo "RUN 1"
# DOES NOT WORK!!
#RUN 1 COMMAND: /opt/mpich-3.3.2/bin/mpirun -np 8 python sn_boilerplate_main.py run --data-parallel --ws 8 --reduce-on-rdu --pef='pef/sn_boilerplate/sn_boilerplate.pef'
#usage: sn_boilerplate_main.py [-h] {compile,run,test,measure-performance} ...
#sn_boilerplate_main.py: error: unrecognized arguments: --ws 8
COMMAND="/opt/mpich-3.3.2/bin/mpirun -np 8 python sn_boilerplate_main.py run --data-parallel --ws 8 --reduce-on-rdu --pef='pef/sn_boilerplate/sn_boilerplate.pef'"
echo "RUN 1 COMMAND: $COMMAND"
#eval $COMMAND

echo "RUN 2"
#ERROR: Unexpected segmentation fault encountered in worker.
COMMAND="/opt/mpich-3.3.2/bin/mpirun -np 1 python sn_boilerplate_main.py run --data-parallel --reduce-on-rdu --pef='pef/sn_boilerplate/sn_boilerplate.pef'"
echo "RUN 2 COMMAND: $COMMAND"
eval $COMMAND



echo "PERF 1"
# The next line is good!!  4347 fps

# [Warning][SAMBA][Default] # If you are measuring performance of data parallel tasks: please explicitly add --ws with world size in CLI.
# WORKS but warning!!
COMMAND="/opt/mpich-3.3.2/bin/mpirun -np 1 python sn_boilerplate_main.py measure-performance --data-parallel --reduce-on-rdu --pef='pef/sn_boilerplate/sn_boilerplate.pef'"
echo "PERF 1 COMMAND: $COMMAND"
eval $COMMAND

echo "PERF 2"
COMMAND="/opt/mpich-3.3.2/bin/mpirun -np 8 python sn_boilerplate_main.py measure-performance --data-parallel --ws 8 --reduce-on-rdu --pef='pef/sn_boilerplate/sn_boilerplate.pef'"
echo "PERF 2 COMMAND: $COMMAND"
#eval $COMMAND

echo "Duration: " $SECONDS
