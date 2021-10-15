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
(venv) username@sm-01:~$ git clone https://git.cels.anl.gov/ai-testbed-apps/sambanova/sambanova_starter.git
Cloning into 'sambanova_starter'...
Username for 'https://github.com': < username >
Password for 'https://username@github.com': < password or PAT >

## Git

Do any necessary git commands.

```bash
(venv) username@sm-01:~$ cd sambanova_starter
(venv) username@sm-01:~/sambanova_starter$ git checkout < branch >
```

Example:

```bash
git checkout feature/100-data-parallel
```

## Change Directory

If you haven't already:

```bash
(venv) username@sm-01:~$ cd sambanova_starter
```

## Commands Arguments

See the folling link for a list of arguments.
[https://confluence.cels.anl.gov/display/AI/SambaNova#SambaNova-Arguments](https://confluence.cels.anl.gov/display/AI/SambaNova#SambaNova-Arguments)

## Sbatch

```bash
sbatch run.sh

sbatch compile_dataparallel.sh
sbatch --gres=rdu:2 run_dataparallel.sh
```
