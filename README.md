# gym-pybullet-drones

This is a TU Delft project based on [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) repository


## Installation


```sh
conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet` and do this command in the directory with pyproject.toml

# check installed packages with `conda list`, deactivate with `conda deactivate`, remove with `conda remove -n drones --all`
```

## Use

### run the project

```sh
cd gym_pybullet_drones/examples/
python3 ro47005.py # position and velocity reference
```
