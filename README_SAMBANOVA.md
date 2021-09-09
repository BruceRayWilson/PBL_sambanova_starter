# Using SambaNova

## Repo

```bash
git clone https://github.com/BruceRayWilson/sambanova_starter
```

## SSH to SambaNova

```bash
<your local machine> $ ssh <ANL username>@homes.cels.anl.gov
username@homes-01:~$ ssh sm-01.cels.anl.gov
username@sm-01.cels.anl.gov's password:
```

## Setup Environment

```bash
username@sm-01:~$ export PATH=$PATH:/opt/sambaflow/bin;export OMP_NUM_THREADS=1;source /opt/sambaflow/venv/bin/activate
```

## Cloning Repo on SambaNova

```bash
username@sm-01:~$ git clone https://github.com/BruceRayWilson/sambanova_starter
Cloning into 'sambanova_starter'...
Username for 'https://github.com': < username >
Password for 'https://username@github.com': < password or PAC >

## Git

Do any necessary git commands.

```bash
[username@medulla1 ~]$ cd sambanova_starter
[username@medulla1 sambanova_starter]$ git checkout < branch >
```

Example:

```bash
git checkout feature/001-start
```

## Change Directory

```bash
# [username@medulla1 ~sambanova_starter]$ cd model/pytorch
or
[username@medulla1 ~]$ cd 3D-PyTorch/model/pytorch
```

## Commands Arguments

See the folling link for a list of arguments.
[https://confluence.cels.anl.gov/display/AI/SambaNova#SambaNova-Arguments](https://confluence.cels.anl.gov/display/AI/SambaNova#SambaNova-Arguments)

## Commands

### Run as multiple files.

```bash
python sn_boilerplate_main.py compile -b=1 --pef-name="sn_boilerplate" --output-folder="pef"
python sn_boilerplate_main.py test --pef="pef/sn_boilerplate/sn_boilerplate.pef"
python sn_boilerplate_main.py run --pef="pef/sn_boilerplate/sn_boilerplate.pef"
python sn_boilerplate_main.py measure-performance --pef="pef/sn_boilerplate/sn_boilerplate.pef"
```
