#!/bin/bash
export OMP_NUM_THREADS=8
source /opt/sambaflow/venv/bin/activate
echo "COMPILE"
python sn_boilerplate_main.py compile --data-parallel -ws 2 -b=1 --pef-name=sn_boilerplate
