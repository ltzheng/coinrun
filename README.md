# CoinRun environment for Genie

This is code for the environments used for minimal reproducible case study in Genie.

## Install

```
# Linux
apt-get install mpich build-essential qt5-default pkg-config
# Mac
brew install qt open-mpi pkg-config

git clone https://github.com/openai/coinrun.git
cd coinrun

conda create -n coinrun python=3.6
conda activate coinrun
pip install tensorflow-gpu==1.12.0
pip install -r requirements.txt
pip install -e .
```

Note that this does not compile the environment, the environment will be compiled when the `coinrun` package is imported.

## Collect 10M transitions

```
python -m coinrun.random_agent --num-levels 10000 --set-seed 42
```
