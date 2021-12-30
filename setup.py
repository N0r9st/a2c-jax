from setuptools import find_packages, setup

setup(
    name='a2c-continious-jax',
    packages=find_packages(),
    version='0.0.1',
    install_requires=['gym', 'mujoco_py', 'flax', 'stable-baselines3', 'wandb'])