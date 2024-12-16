# Polynomial Differential Networks for End-to-End Autonomous Driving


## Setup

```Shell
git clone https://github.com/choys0401/polydiff.git
cd polydiff
sh setup_carla.sh
sh download_ckpt.sh
conda env create -f environment.yml
conda activate polydiff
```

## Run CARLA

```Shell
sh carla/CarlaUE4.sh --world-port=20000 &
```
or
```Shell
sh carla/CarlaUE4.sh --world-port=20000 -opengl &
```

## Evaluation

```Shell
sh run_evaluation.sh
```
