#!/bin/bash
# Stop on error
set -e
# Print commands
set -x

#######################
# Run script
# sbatch --gres=rdu:2 run_dataparallel.sh
#######################
# Edit these variables.
#######################
MODEL_NAME="FFNLogReg"
#######################
# Start script timer
SECONDS=0

echo "Model: ${MODEL_NAME}"
echo "Date: " $(date +%m/%d/%y)
echo "Time: " $(date +%H:%M)

# Change the fully qualified directory path as necessary.

echo "COMPILE Separately!!"

echo "RUN"
/opt/mpich-3.3.2/bin/mpirun -np 2 python /homes/wilsonb/sambanova_starter_ANL/sn_boilerplate_main.py run --data-parallel --reduce-on-rdu --pef=/homes/wilsonb/sambanova_starter_ANL/pef/sn_boilerplate/sn_boilerplate.pef

echo "PERF"
/opt/mpich-3.3.2/bin/mpirun -np 2 python sn_boilerplate_main.py measure-performance --data-parallel --reduce-on-rdu --pef=/homes/wilsonb/sambanova_starter_ANL/pef/sn_boilerplate/sn_boilerplate.pef

echo "Duration: " $SECONDS
