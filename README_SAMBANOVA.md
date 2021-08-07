# Using SambaNova

## Notes

```text
I use common_app_driver which covers everything: from sambaflow.samba.utils.common import common_app_driver #rcw uncommented
common_app_driver(args=args,
                      model=model,
                      inputs=inputs,
                      name=name,
                      optim=optim,
                      squeeze_bs_dim=True,
                      get_output_grads=False,
                      app_dir=utils.get_file_dir(__file__))
```

## Running Original Code

```bash
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml
```

## Repo

```bash
git clone https://github.com/tanwimallick/dcrnn_pytorch/
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
username@sm-01:~$ git clone https://github.com/tanwimallick/dcrnn_pytorch/
Cloning into 'dcrnn_pytorch'...
Username for 'https://github.com': < username >
Password for 'https://username@github.com': < password >

## Git

Do any necessary git commands.

```bash
[username@medulla1 ~]$ cd dcrnn_pytorch
[username@medulla1 dcrnn_pytorch]$ git checkout < branch >
```

Example:

```bash
git checkout feature/001-start
```

## Change Directory

```bash
# [username@medulla1 ~dcrnn_pytorch]$ cd model/pytorch
or
[username@medulla1 ~]$ cd 3D-PyTorch/model/pytorch
```

## Commands Arguments

See the folling link for a list of arguments.
[https://confluence.cels.anl.gov/display/AI/SambaNova#SambaNova-Arguments](https://confluence.cels.anl.gov/display/AI/SambaNova#SambaNova-Arguments)

## Commands

```bash
python dcrnn.py compile --config_filename="./data/model/dcrnn_test_config.yaml" -b=1 --pef-name="dcrnn" --output-folder="pef"
python dcrnn.py test --pef="pef/dcrnn/dcrnn.pef"
python dcrnn.py run --config_filename="./data/model/dcrnn_test_config.yaml" --pef="pef/dcrnn/dcrnn.pef"
python dcrnn.py measure-performance --pef="pef/dcrnn/dcrnn.pef"
```
