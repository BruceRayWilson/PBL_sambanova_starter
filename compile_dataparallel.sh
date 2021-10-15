#!/bin/bash
export OMP_NUM_THREADS=8
source /opt/sambaflow/venv/bin/activate
#######################
# Run script
# sbatch compile_dataparallel.sh
#######################
# Start script timer
SECONDS=0

echo "COMPILE"
python sn_boilerplate_main.py compile --data-parallel -ws 2 -b=1 --pef-name=sn_boilerplate --output-folder="pef"

echo "Duration: " $SECONDS
