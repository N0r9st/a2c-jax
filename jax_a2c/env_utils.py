import functools
import multiprocessing as mp
from typing import Iterable, List, Optional, Tuple

import gym
import numpy as np
from mujoco_py import MjSimState
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


def _worker(remote, parent_remote, env_fn) -> None:

    parent_remote.close()
    env = env_fn()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                if done:
                    info["terminal_observation"] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "set":
                env.sim.set_state(data)
                remote.send(None)
            elif cmd == "get_state":
                remote.send(env.sim.get_state())
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            print('PROCESS ENDED')
            break

class SubprocVecEnv:

    def __init__(self, env_fns, start_method=None):

        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, env_fn)
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
        self.num_envs = len(self.processes)
        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def step_async(self, actions) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> Tuple[np.array]:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self) -> np.array:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return _flatten_obs(obs)

    def step(self, actions: Iterable) -> Tuple[np.array]:
        self.step_async(actions)
        return self.step_wait()
        
    def set_state(self, env_states: Iterable) -> None:
        for remote, env_state in zip(self.remotes, env_states):
            remote.send(("set", env_state))
        [remote.recv() for remote in self.remotes]

    def get_state(self,):
        for remote in self.remotes:
            remote.send(("get_state", None))
        return [remote.recv() for remote in self.remotes]
    def getattr_depth_check(self, *args, **kwargs):
        return None

def _flatten_obs(obs: List[np.array]) -> np.array:
    stacked = np.stack(obs)
    return stacked


def create_env(name: str = 'HalfCheetah-v3', env_state: Optional[MjSimState] = None, seed=None):
    env = gym.make(name)
    env.reset()
    if env_state:
        env.sim.set_state(env_state)
    if seed is not None:
        env.seed(seed)
    return env

def make_env_fn(name: str = 'HalfCheetah-v3', env_state: Optional[MjSimState] = None, seed=None):
    env_fn = functools.partial(create_env, name=name, env_state=env_state, seed=seed)
    return env_fn

def make_vec_env(
    name: str = 'HalfCheetahEnv', 
    env_state: Optional[MjSimState] = None, 
    num: int = 4, norm_r=True, 
    norm_obs=True, 
    seed=None):
    if seed is None:
        env_func_list = [make_env_fn(name=name, env_state=env_state, seed=seed) for _ in range(num)]
    else:
        env_func_list = [make_env_fn(name=name, env_state=env_state, seed=seed+i) for i in range(num)]
    return VecNormalize(SubprocVecEnv(env_func_list), norm_obs=norm_obs, norm_reward=norm_r)
