# Contrastive Intrinsic Control (CIC)

This is the official codebase for [CIC: Contrastive Intrinsic Control for Unsupervised Skill Discovery](https://arxiv.org/abs/2202.00161). If you find this codebase useful to your research you can cite it as:

```
@misc{laskin2022cic,
Author = {Michael Laskin and Hao Liu and Xue Bin Peng and Denis Yarats and Aravind Rajeswaran and Pieter Abbeel},
Title = {CIC: Contrastive Intrinsic Control for Unsupervised Skill Discovery},
Year = {2022},
Eprint = {arXiv:2202.00161},
}
```

This codebase is built on top of the [Unsupervised Reinforcement Learning Benchmark (URLB) codebase](https://anonymous.4open.science/r/urlb). We include agents for all baselines in the `agents` folder. Our method `CIC`  is implemented in `agents/cic.py` and the config is specified in `agents/cic.yaml`.

To pre-train CIC, run the following command:

```sh
python pretrain.py agent=cic domain=walker experiment=YOUR_EXP_NAME
```

To finetune CIC, run the following command. Make sure to specify the directory of your saved snapshots with `YOUR_EXP_NAME`.

```sh
python finetune.py pretrained_agent=cic agent=cic experiment=YOUR_EXP_NAME task=walker_stand snapshot_ts=2000000 agent=cic
```

## Requirements
We assume you have access to a GPU that can run CUDA 10.2 and CUDNN 8. Then, the simplest way to install all required dependencies is to create an anaconda environment by running
```sh
conda env create -f conda_env.yml
```
After the instalation ends you can activate your environment with
```sh
conda activate urlb
```

## Available Domains
We support the following domains.
| Domain | Tasks |
|---|---|
| `walker` | `stand`, `walk`, `run`, `flip` |
| `quadruped` | `walk`, `run`, `stand`, `jump` |
| `jaco` | `reach_top_left`, `reach_top_right`, `reach_bottom_left`, `reach_bottom_right` |


### Monitoring
Logs are stored in the `exp_local` folder. To launch tensorboard run:
```sh
tensorboard --logdir exp_local
```
The console output is also available in a form:
```
| train | F: 6000 | S: 3000 | E: 6 | L: 1000 | R: 5.5177 | FPS: 96.7586 | T: 0:00:42
```
a training entry decodes as
```
F  : total number of environment frames
S  : total number of agent steps
E  : total number of episodes
R  : episode return
FPS: training throughput (frames per second)
T  : total training time
```
