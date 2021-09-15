#!/bin/bash
# Stop on error -not
set -x
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

#rcw export SF_RNT_TILE_AFFINITY=0xf0000000

echo "Model: ${MODEL_NAME}"
echo "Date: " $(date +%m/%d/%y)
echo "Time: " $(date +%H:%M)

# Change this directory path as necessary.

echo "COMPILE Seperately!!"

echo "RUN"
/opt/mpich-3.3.2/bin/mpirun -hosts sm-02 -np 2 python /homes/ac.rick.weisner/tmp/wilsonb/sambanova_starter/sn_boilerplate_main.py run --data-parallel --reduce-on-rdu --pef=/homes/ac.rick.weisner/tmp/wilsonb/sambanova_starter/out/sn_boilerplate/sn_boilerplate.pef

echo "PERF"
# The next line is good!!  4347 fps

# [Warning][SAMBA][Default] # If you are measuring performance of data parallel tasks: please explicitly add --ws with world size in CLI.
# WORKS but warning!!
/opt/mpich-3.3.2/bin/mpirun -hosts sm-02 -np 2 python sn_boilerplate_main.py measure-performance --data-parallel --reduce-on-rdu --pef=/homes/ac.rick.weisner/tmp/wilsonb/sambanova_starter/out/sn_boilerplate/sn_boilerplate.pef

echo "Duration: " $SECONDS
