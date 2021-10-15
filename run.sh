#!/bin/sh
#######################
# Run script
# sbatch run.sh
#######################

# Change this directory path as necessary.
cd ~/sambanova_starter
python sn_boilerplate_main.py compile -b=1 --pef-name="sn_boilerplate" --output-folder="pef"
python sn_boilerplate_main.py test --pef="pef/sn_boilerplate/sn_boilerplate.pef"
python sn_boilerplate_main.py run --pef="pef/sn_boilerplate/sn_boilerplate.pef"
python sn_boilerplate_main.py measure-performance --pef="pef/sn_boilerplate/sn_boilerplate.pef"
