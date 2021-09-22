#!/bin/bash
# Stop on error
set -e
# Print commands
set -x

#######################
# Edit these variables.
#######################
MODEL_NAME="FFNLogReg"
#######################
# Start script timer
SECONDS=0

export SF_RNT_TILE_AFFINITY=0xff000000

echo "Model: ${MODEL_NAME}"
echo "Date: " $(date +%m/%d/%y)
echo "Time: " $(date +%H:%M)

# Change the fully qualified directory path as necessary.

echo "COMPILE Seperately!!"

echo "RUN"
/opt/mpich-3.3.2/bin/mpirun -hosts sm-02 -np 2 python /homes/wilsonb/sambanova_starter/sn_boilerplate_main.py run --data-parallel --reduce-on-rdu --pef=/homes/wilsonb/sambanova_starter/out/sn_boilerplate/sn_boilerplate.pef

echo "PERF"
/opt/mpich-3.3.2/bin/mpirun -hosts sm-02 -np 2 python sn_boilerplate_main.py measure-performance --data-parallel --reduce-on-rdu --pef=/homes/wilsonb/sambanova_starter/out/sn_boilerplate/sn_boilerplate.pef

echo "Duration: " $SECONDS
