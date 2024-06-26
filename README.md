# Practical Reinforcement Learning using Time-effieicent Model-based Policy Optimization

Code to reproduce the experiments in Practical Reinforcement Learning using Time-effieicent Model-based Policy Optimization. This paper is currently submitted to IEEE Transactions on Automation Science and Engineering (T-ASE) for peer review.

![method.png](https://raw.githubusercontent.com/mrjun123/PMBPO/main/images/method.png)

Please feel free to contact us regarding to the details of implementing PMBPO. (Wenjun Huang: wj.huang1@siat.ac.cn Yunduan Cui: cuiyunduan@gmail.com)
## Requirements

1. Install [MuJoCo 1.31](https://www.roboti.us/download.html) and copy your [license](https://www.roboti.us/file/mjkey.txt) key to `~/.mujoco/mjkey.txt`. 
2. Install [PyTorch](https://pytorch.org/get-started/previous-versions/), we recommend CUDA 11.6 and Pytorch 1.13.1
3. Other dependencies can be installed with `pip install -r requirements.txt`.

## Running Experiments

Experiment for a specific configuration can be run using:

```python
python main.py --config half_cheetah
```

The specific configuration file is located in the `configs` directory and the default configuration file can be located in the root directory `default_config.json` was found, which allows you to modify the experimental parameters.

## Logging

We use Tensorboard to record experimental data, you can view runs with:

```python
tensorboard --logdir ./runs/ --port=6006 --host=0.0.0.0
```

