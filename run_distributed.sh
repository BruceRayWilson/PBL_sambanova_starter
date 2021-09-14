#!/bin/sh

alias snpath='export PATH=$PATH:/opt/sambaflow/bin'
alias snthreads='export OMP_NUM_THREADS=1'
alias snvenv='source /opt/sambaflow/venv/bin/activate'
alias snp='snpath;snthreads;snvenv'

snp

# Change this directory path as necessary.
cd ~/sambanova_starter

# ws = world_size and its value really doesn't matter for a compile.
python sn_boilerplate_main.py compile --data-parallel -ws 2 -b=1 --pef-name="sn_boilerplate" --output-folder="pef"

#python sn_boilerplate_main.py test --distributed-run --pef="pef/sn_boilerplate/sn_boilerplate.pef"

#python sn_boilerplate_main.py run --distributed-run --pef="pef/sn_boilerplate/sn_boilerplate.pef"
/opt/mpich-3.3.2/bin/mpirun -np 8 python sn_boilerplate_main.py run --data-parallel --reduce-on-rdu --pef="pef/sn_boilerplate/sn_boilerplate.pef"

#python sn_boilerplate_main.py measure-performance --distributed-run --pef="pef/sn_boilerplate/sn_boilerplate.pef"
/opt/mpich-3.3.2/bin/mpirun -np 8 python sn_boilerplate_main.py measure-performance --data-parallel -ws 8 --reduce-on-rdu --pef="pef/sn_boilerplate/sn_boilerplate.pef"

